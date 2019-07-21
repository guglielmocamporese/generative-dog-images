################################
# Imports
################################

import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt

# Custom imports
from utils import time2str
from params import *


################################
# Define the GAN object
################################

class GAN(object):
    def __init__(self, noise_dim=100, image_dim=64, name='gan', debug=False):
        self.noise_dim, self.image_dim = noise_dim, image_dim
        self.name = name
        self.debug = debug
        self.noise = tf.placeholder(tf.float32, [None, self.noise_dim])
        self.image = tf.placeholder(tf.float32, [None, self.image_dim, self.image_dim, 3])
        self.is_training = tf.placeholder(tf.bool)
        self.epochs_time = []
        with tf.variable_scope(self.name):
            self.G_logits = self.get_G_logits(self.noise, train=self.is_training)
            self.D_true_logits = self.get_D_logits(self.image)
            self.D_fake_logits = self.get_D_logits(self.G_logits, reuse=True)
            self.D_loss, self.G_loss, self.disc_step, self.gen_step = self.model_compile()
        
    def get_G_logits(self, x, reuse=False, train=True):
        with tf.variable_scope('generator'.format(self.name), reuse=reuse):
            
            # 4x4x512
            x = tf.layers.dense(x, 4 * 4 * 512, use_bias=False)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.leaky_relu(x)
            x = tf.reshape(x, (-1, 4, 4, 512))

            # 4x4x512 -> 8x8x256
            x = tf.layers.conv2d_transpose(x, 256, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV), use_bias=False)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.leaky_relu(x)

            # 8x8x256 -> 16x16x128
            x = tf.layers.conv2d_transpose(x, 128, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV), use_bias=False)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.leaky_relu(x)

            # 16x16x128 -> 32x32x64
            x = tf.layers.conv2d_transpose(x, 64, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV), use_bias=False)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.leaky_relu(x)

            # 32x32x64 -> 64x64x32
            x = tf.layers.conv2d_transpose(x, 32, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV), use_bias=False)
            x = tf.layers.batch_normalization(x, training=train)
            x = tf.nn.leaky_relu(x)

            # 64x64x32 -> 64x64x3
            x = tf.layers.conv2d_transpose(x, 3, 5, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
            x = tf.tanh(x)
            return x
        
    def get_D_logits(self, x, reuse=False):
        with tf.variable_scope('discriminator'.format(self.name), reuse=reuse):
            
            # 64x64x3 -> 32x32x32
            x = tf.layers.conv2d(x, 32, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.3)

            # 32x32x32 -> 16x16x64
            x = tf.layers.conv2d(x, 64, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.3)

            # 16x16x64 -> 8x8x128
            x = tf.layers.conv2d(x, 128, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.3)

            # 8x8x128 -> 8x8x256
            x = tf.layers.conv2d(x, 256, 5, padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.leaky_relu(x)
            x = tf.layers.dropout(x, 0.3)

            # 8x8x256 -> 4x4x128
            x = tf.layers.conv2d(x, 128, 5, strides=(2, 2), padding='same', kernel_initializer=tf.truncated_normal_initializer(stddev=WEIGHT_INIT_STDDEV))
            x = tf.layers.batch_normalization(x, training=True)
            x = tf.nn.leaky_relu(x)

            x = tf.reshape(x, (-1, 4 * 4 * 128))
            x = tf.layers.dense(x, 1)
            return x
        
    def model_compile(self):
        
        # Define the loss and the optimization procedure
        D_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_true_logits, labels=tf.ones_like(self.D_true_logits)) + tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.zeros_like(self.D_fake_logits)))
        G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_fake_logits, labels=tf.ones_like(self.D_fake_logits)))
        
        # Retrieve the variables to backpropagate
        disc_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}/discriminator'.format(self.name))
        gen_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='{}/generator'.format(self.name))
    
        # Train step for D and G
        disc_step = tf.train.AdamOptimizer(learning_rate=LR_D, beta1=0.5).minimize(D_loss, var_list=disc_vars)
        gen_step = tf.train.AdamOptimizer(learning_rate=LR_G, beta1=0.5).minimize(G_loss, var_list=gen_vars)
        
        return D_loss, G_loss, disc_step, gen_step
        
    def train(self, feed, epochs=1):

        def get_iterations(epoch):
            D_iter = 5
            G_iter = 1
            return D_iter, G_iter
        
        # Saver for saving the model
        self.saver = tf.train.Saver()
            
        with tf.Session() as sess:
            
            # Init the variables
            sess.run(tf.global_variables_initializer())

            # Epochs
            for e in range(1, epochs + 1):
                
                # Time
                t0 = time.time()

                # Shuffle the data
                feed.shuffle_data()

                # Mini-batches propagation and back-propagation
                D_loss_train, G_loss_train = [], []
                D_iter, G_iter = get_iterations(e)
                while feed.has_more_data:
                    for _ in range(D_iter):
                        if not feed.has_more_data:
                            break

                        image_batch = feed.get_batch(augment=True, p_aug=0.5)
                        noise_batch = np.random.uniform(-1, 1, size=np.prod([len(image_batch), self.noise_dim])).reshape([len(image_batch), self.noise_dim])
                        _, D_loss_batch = sess.run([self.disc_step, self.D_loss], feed_dict={self.noise: noise_batch, self.image: image_batch, self.is_training: True})
                        D_loss_train.append(D_loss_batch)

                    for _ in range(G_iter):
                        if not feed.has_more_data:
                            break

                        image_batch = feed.get_batch()
                        noise_batch = np.random.uniform(-1, 1, size=np.prod([len(image_batch), self.noise_dim])).reshape([len(image_batch), self.noise_dim])
                        _, G_loss_batch = sess.run([self.gen_step, self.G_loss], feed_dict={self.noise: noise_batch, self.is_training: True})
                        G_loss_train.append(G_loss_batch)
                
                # Time
                self.epochs_time.append(time.time() - t0)

                # Reset the data
                feed.reset()

                # Print train info
                str_template = 'Epoch: {}, D Train Loss: {}, G Train Loss: {}, Epoch Time: {}, Time to End: {}'
                print(str_template.format(
                                        e, 
                                        np.array(D_loss_train).mean(), np.array(G_loss_train).mean(), 
                                        time2str(self.epochs_time[-1]), 
                                        time2str(np.array(self.epochs_time).mean() * (epochs - e))
                                    ))
                
                # View generation
                if e % 10 == 0:
                    images_gen = self.generate(num_images=64, seed=1234)

                    # Plot generated images
                    print('\tGeneration')
                    plt.figure(figsize=[20, 20])
                    for idx_image, image in enumerate(images_gen):
                        plt.subplot(int(np.ceil(np.sqrt(len(images_gen)))), int(np.ceil(np.sqrt(len(images_gen)))), idx_image + 1)
                        plt.imshow(image)
                        plt.xticks([])
                        plt.yticks([])
                    plt.show()

                # Save the model
                self.saver.save(sess, self.name)

        print('Train finished!')
        
    def generate(self, noise=None, num_images=16, seed=None):

        # Sample randomly
        if noise is None:
            if seed:
                np.random.seed(seed)
            noise = np.random.uniform(-1, 1, size=np.prod([num_images, self.noise_dim])).reshape([num_images, self.noise_dim])
    
        #saver = tf.train.Saver()
        with tf.Session() as sess:
            
            self.saver.restore(sess, './{}'.format(self.name))
            
            # Compute the inference
            G_out = sess.run(tf.nn.sigmoid(self.G_logits), feed_dict={self.noise: noise, self.is_training: False})
            return G_out