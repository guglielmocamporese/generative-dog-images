################################
# Imports
################################

import os
import numpy as np
import cv2
from tqdm import tqdm
import math

# Custom imports
from utils import parse_xml


################################
# Feeder object
################################

class Feeder(object):
    def __init__(self, image_base_folder, attr_base_folder, batch_size=64, seed=1234):
        self.image_base_folder = image_base_folder
        self.attr_base_folder = attr_base_folder
        self.batch_size = batch_size
        self.seed = seed
        np.random.seed(self.seed)
        self.xml_paths = self.get_xml_paths()
        self.images = self.load_images()
        self.num_images = len(self.images)
        self.idx_data = [i for i in range(self.num_images)]
        self.idx_batch = 0
        self.num_batches = math.ceil(self.num_images / self.batch_size)
        self.has_more_data = self.num_images > 0
        
    def get_xml_paths(self):
        xml_paths = []
        for dog_class in sorted(os.listdir(self.attr_base_folder)):
            if dog_class == '.DS_Store':
                continue
            for xml_name in sorted(os.listdir(os.path.join(self.attr_base_folder, dog_class))):
                if xml_name == '.DS_Store':
                    continue
                xml_path = os.path.abspath(os.path.join(self.attr_base_folder, dog_class, xml_name))
                xml_paths.append(xml_path)
        return xml_paths
    
    def load_images(self):
        images = []
        for xml_path in tqdm(self.xml_paths, desc='Loading images'):
            
            # Parse xml image attributes
            xml_dict = parse_xml(xml_path)
            
            # Load and preprocess image
            try:
                image_path = os.path.join(self.image_base_folder, '{}.jpg'.format(xml_dict['filename']))
                image = cv2.imread(image_path, -1)
                image = self.preporcess_image(image, xml_dict)
                images.append(image)
            except:
                continue
            
        images = np.array(images)
        return images
        
    def get_batch(self, augment=False, p_aug=0.5):
            
        # Load the batch
        images = []
        for idx in self.idx_data[self.idx_batch * self.batch_size: (self.idx_batch + 1) * self.batch_size]:
            
            # Load and preprocess the image
            image = self.images[idx]
            
            # Augment image
            if augment:
                if np.random.rand() >= p_aug:
                    image = self.augment_image(image)
            images.append(image)
        images = np.array(images)
            
        # Update the batch index
        if self.idx_batch < self.num_batches - 1:
            self.idx_batch += 1
        else:
            self.has_more_data = False
        return images
            
    def shuffle_data(self):
        np.random.shuffle(self.idx_data)
            
    def reset(self):
        self.idx_batch = 0
        self.has_more_data = len(self.xml_paths) > 0
    
    def preporcess_image(self, image, xml_dict):
        
        # Crop image
        x_min, x_max = int(xml_dict['xmin']), int(xml_dict['xmax'])
        y_min, y_max = int(xml_dict['ymin']), int(xml_dict['ymax'])
        image = image[y_min:y_max, x_min:x_max, :]
        
        # Resize image
        image = cv2.resize(image, (64, 64))
        
        # bgr2rgb
        image = np.stack([image[:, :, 2], image[:, :, 1], image[:, :, 0]], axis=-1)
        
        # Normalize image
        image = (image / 255. - 0.5) * 2.0
        return image
    
    def augment_image(self, image):
        image_aug = np.fliplr(image)
        return image_aug