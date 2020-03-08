import os
import glob

import numpy as np
from PIL import Image
import chainer.links
from chainer.datasets import tuple_dataset
from chainer import serializers
import matplotlib.pyplot as plt

import model

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../face_predictor/models/")


class GenderPredict:
    def __init__(self, persons_path):
        self.model_path = "{}/etc/AlexlikeMSGD.model".format(MODEL_PATH)
        self.persons_path = persons_path
        self.female_path = self.persons_path + "females/"
        self.male_path = self.persons_path + "males/"
        self.chainer_model = chainer.links.Classifier(model.Alex())

    def generate_dataset(self):
        """
        男女推定用データセットの作成
        :return:
        """
        images = []
        labels = []

        # 画像の読み込み
        image_files = glob.glob(self.persons_path + "*.png")
        print(image_files)
        for image_file in image_files:
            image = Image.open(image_file)
            print(image)
            try:
                transposed_image = np.asarray(image).transpose((2, 0, 1)).astype(np.float32) / 255.
            except ValueError:
                print("value Error")
                continue

            images.append(transposed_image)
            labels.append(np.int32(0))

        return tuple_dataset.TupleDataset(images, labels)

    def calc_judge(self):
        """
        男女識別の実行
        :return: (male,female)
        """
        class_names = ['女', '男']
        serializers.load_npz(self.model_path, self.chainer_model)
        dataset = self.generate_dataset()

        male_count = 0
        female_count = 0
        print(dataset)
        for x, t in dataset:
            self.chainer_model.to_cpu()
            y = self.chainer_model.predictor(x[None, ...]).data.argmax(axis=1)[0]
            print("Prediction:", class_names[y])
            if y == 0:
                female_image_name = "female_%02d.png" % female_count
                plt.imsave(self.female_path + female_image_name, x.transpose(1, 2, 0))
                female_count += 1
            else:
                male_image_name = "male_%02d.png" % male_count
                plt.imsave(self.male_path + male_image_name, x.transpose(1, 2, 0))
                male_count += 1

        return male_count, female_count
