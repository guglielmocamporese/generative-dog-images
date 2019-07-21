################################
# Imports
################################

import os
import numpy as np
import cv2
from zipfile import ZipFile
from tqdm import tqdm

# Custom imports
from params import *
from feeder import Feeder
from model import GAN


################################
# Main
################################

if __name__ == '__main__':
	
	# Create and train the GAN 
	model = GAN(noise_dim=NOISE_DIM, image_dim=IMAGE_DIM, name='gan', debug=False)
	feed = Feeder(IMAGES_BASE_FOLDER, LABEL_BASE_FOLDER, batch_size=BATCH_SIZE)
	model.train(feed, epochs=EPOCHS)

	# Save submission
	with ZipFile(OUTPUT_ZIP_NAME, 'w') as zip:
		batch_size = 100
		num_batches = int(10000 / batch_size)
		for idx_batch in tqdm(range(num_batches), desc='Writing test images'):
		    images_gen = (model.generate(num_images=batch_size, seed=idx_batch + 1) * 255).astype(np.uint8)
		    for idx_image, image_gen in enumerate(images_gen):
		        image_name = '{}.png'.format((idx_batch + 1) * batch_size + idx_image)
		        cv2.imwrite(image_name, image_gen)
		        zip.write(image_name)
		        os.remove(image_name)
