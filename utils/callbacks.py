import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

class SaveImages(keras.callbacks.Callback):
    """
    This is a subclass of the keras.callbacks.Callback class.
    On subclassing it we can specify methods which can be executed while training
    """

    def __init__(self, noise, margin, num_rows, num_cols, **kwargs):
        super(keras.callbacks.Callback,self).__init__(**kwargs)
        self.noise = noise
        self.margin = margin
        self.num_rows = num_rows
        self.num_cols = num_cols

    # overwriting on_epoch_end() helps in executing a custom method when an epoch ends
    def on_epoch_end(self, epoch, logs=None):
        """
        Saves images generated from a fixed random vector by the generator to the disk 
        
        Parameters:
            noise: fixed noise vector from a normal distribution to be fed to the generator.
            num_rows: number of rows of images
            num_cols: number of columns of images
            margin: margin between images
            generator: keras model representing the generator network
        
        """

        # Generate a base array upon which images can then be added sequentially
        image_array = np.full((
            self.margin + (self.num_rows * (28 + self.margin)),
            self.margin + (self.num_cols * (28 + self.margin)), 3),
            255, dtype=np.uint8)

        # Generate num_rows*num_cols number of images using the generator model
        generated_images = self.model.generator.predict(self.noise, verbose=0)

        # Convert pixel intensities to the range [0,255] from [-1,1]
        generated_images = (generated_images + 1.0) * 127.5

        #Images need not be converted into the typical [0,255] pixel intensity values because the PIL Image module accepts the range [0,1] 
        
        image_count = 0
        for row in range(self.num_rows):
            for col in range(self.num_cols):
                r = row * (28 + 16) + self.margin
                c = col * (28 + 16) + self.margin
                image_array[r:r + 28, c:c + 28] = generated_images[image_count]
                image_count += 1

        # The image array now contains all the images in an array format which can be stored to the disk

        output_path = 'epoch_images'
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        filename = os.path.join(output_path, f"train-{epoch}.png")
        im = Image.fromarray(image_array)
        im.save(filename)