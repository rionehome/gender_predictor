import os
import glob

import rclpy
from rclpy.node import Node
import numpy as np
import chainer.links
from chainer.datasets import tuple_dataset
from chainer import serializers
import matplotlib.pyplot as plt
from rione_msgs.msg import PredictResult
from sensor_msgs.msg import Image

import alex

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../gender_predictor/models/")


class GenderPredict(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.model_path = "{}/etc/AlexlikeMSGD.model".format(MODEL_PATH)
        self.alex = chainer.links.Classifier(alex.Alex())
        self.create_subscription(Image, "/gender_predictor/color/image", self.callback_image, 1)
        self.pub_result = self.create_publisher(PredictResult, "/face_predictor/result", 10)

    def callback_image(self, msg: Image):
        """
        画像のsubscribe
        :param msg:
        :return:
        """
        if not len(msg.data) == 96 ** 2 * 3:
            print("画像サイズは96×96×3である必要があります", flush=True)
            return
        image_array = np.asarray(msg.data).reshape((96, 96, 3))
        y = self.alex.predictor(image_array)
        print(y)

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
        serializers.load_npz(self.model_path, self.alex)
        dataset = self.generate_dataset()

        male_count = 0
        female_count = 0
        print(dataset)
        for x, t in dataset:
            self.alex.to_cpu()
            y = self.alex.predictor(x[None, ...]).data.argmax(axis=1)[0]
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


def main():
    rclpy.init()
    node = GenderPredict("GenderPredictor")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
