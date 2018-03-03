from dataloaders.dataloader import DataLoader
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from matplotlib import pyplot as plt
import numpy as np
import os
from PIL import Image
from scipy import ndimage

DATA_DIR = 'data/imagenet/bird'
HEIGHT = 32
WIDTH = 32
CHANNELS = 3


class ImagenetDataLoader(DataLoader):
    def __init__(self, batch_size: int) -> None:
        super().__init__('imagenet', HEIGHT, WIDTH, CHANNELS, batch_size)
        self.img_dir = os.path.join(os.getcwd(), DATA_DIR)
        self.datagen = ImageDataGenerator()

    def load_data(self) -> None:
        """Load the dataset."""
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError('Image dir does not exist')

        datagen = self.datagen.flow_from_directory(
            self.img_dir,
            target_size=(HEIGHT, WIDTH),
            class_mode=None,
            shuffle=True,
            batch_size=self.batch_size
        )
        self.x_train = np.concatenate(datagen)

        # image_links = [os.path.join(self.img_dir, img) for img in os.listdir(image_dir)]
        # images = []
        # for img_link in image_links:
        #     img = img_to_array(load_img(img_link))
        #     herp = (1,) + img.shape
        #     img = img.reshape((1,) + img.shape)
        #     # img = img.reshape((1, HEIGHT, WIDTH, CHANNELS))
        #     images.append(img)
        #
        # self.x_train = np.array(images)
        # batch = self.get_batches(self.img_dir)
        # derp = 1
        # return batch

    def next_batch(self):
        """Fetch the next batch of images from the dataset."""
        stop = self.batch_num + self.batch_size
        data = self.x_train[self.batch_num:stop]

        self.batch_num += self.batch_size
        if self.batch_num > len(self.x_train) - self.batch_size:
            self.batch_num = 0

        return data

    def view_image(self):
        batches = self.get_batches(self.img_dir)
        batch = next(batches)
        plt.imshow(batch[0][0])
