from datetime import datetime
import os

import keras
from keras import models
from keras import layers
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from lib.utils import save_history, show_image_tile

os.chdir(os.path.dirname(os.path.abspath(__file__)))

DATASET_PATH = "./data/"
LOG_PATH = os.path.join("log/", datetime.now().strftime('%Y%m%d_%H%M%S'))

IMAGE_SIZE = 96
BATCH_SIZE = 100
NUM_EPOCH = 100


def create_model():
    input_data = layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    conv1 = layers.Conv2D(48, kernel_size=(3, 3), activation='relu')(input_data)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.MaxPool2D((2, 2))(conv1)
    conv2 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.MaxPool2D((2, 2))(conv2)
    conv3 = layers.Conv2D(192, kernel_size=(3, 3), activation='relu')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv4 = layers.Conv2D(192, kernel_size=(3, 3), activation='relu')(conv3)
    conv4 = layers.BatchNormalization()(conv4)
    conv5 = layers.Conv2D(128, kernel_size=(3, 3), activation='relu')(conv4)
    conv5 = layers.BatchNormalization()(conv5)
    conv5 = layers.MaxPool2D((2, 2))(conv5)
    flatten = layers.Flatten()(conv5)
    dense1 = layers.Dense(1024)(flatten)
    dense2 = layers.Dense(1024)(dense1)
    predict = layers.Dense(2, activation='softmax')(dense2)
    model = models.Model(inputs=input_data, outputs=predict)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizers.RMSprop(lr=1e-4),
        metrics=['acc']
    )

    return model


def train():
    data_gen = ImageDataGenerator(rescale=1. / 255)
    train_data_iterator = data_gen.flow_from_directory(
        os.path.join(DATASET_PATH, "train_image"),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    val_data_iterator = data_gen.flow_from_directory(
        os.path.join(DATASET_PATH, "val_image"),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )
    test_data_iterator = data_gen.flow_from_directory(
        os.path.join(DATASET_PATH, "test_image"),
        target_size=(IMAGE_SIZE, IMAGE_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical"
    )

    model = create_model()
    model.summary()

    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_acc',
            patience=10
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(LOG_PATH, "model.h5"),
            monitor="val_loss",
            save_best_only=True
        ),
        keras.callbacks.TensorBoard(
            log_dir=os.path.join(LOG_PATH, "tensor_board_log"),
            histogram_freq=0
        )
    ]

    history = model.fit_generator(
        train_data_iterator,
        steps_per_epoch=1000,
        epochs=NUM_EPOCH,
        validation_data=val_data_iterator,
        validation_steps=50,
        callbacks=callbacks_list
    )

    save_history(history, LOG_PATH)

    model = models.load_model(os.path.join(LOG_PATH, "model.h5"))

    print(model.evaluate_generator(test_data_iterator))
    print(model.predict_generator(test_data_iterator))


if __name__ == '__main__':
    train()
