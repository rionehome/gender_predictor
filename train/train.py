import os

import matplotlib.pyplot as plt

from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = "../../data/kaggle_dogs_vs_cats/"
IMAGE_SIZE = 96
BATCH_SIZE = 32
NUM_EPOCH = 100


def train():
    data_gen = ImageDataGenerator(rescale=1. / 255)
    validation_data_iterator = data_gen.flow_from_directory(
        DATASET_PATH + "validation",
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="binary"
    )


if __name__ == '__main__':
    train()
