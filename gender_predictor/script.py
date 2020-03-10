import os

import rclpy
from rclpy.node import Node
import numpy as np
import chainer.links
from chainer import serializers
from rione_msgs.msg import PredictResult
from sensor_msgs.msg import Image
from std_msgs.msg import String

import alex

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../gender_predictor/models/")


class GenderPredict(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.model_path = "{}/AlexlikeMSGD.model".format(MODEL_PATH)
        self.alex = chainer.links.Classifier(alex.Alex())
        self.create_subscription(Image, "/gender_predictor/color/image", self.callback_image, 1)
        self.pub_result = self.create_publisher(PredictResult, "/gender_predictor/result", 10)

    def callback_image(self, msg: Image):
        """
        画像のsubscribe
        :param msg:
        :return:
        """
        if not len(msg.data) == 96 ** 2 * 3:
            print("画像サイズは96×96×3である必要があります", flush=True)
            return
        image_array = np.asarray(msg.data).reshape((1, 96, 96, 3)).transpose((0, 3, 1, 2)).astype(np.float32) / 255.
        serializers.load_npz(self.model_path, self.alex)
        self.alex.to_cpu()
        with chainer.using_config('train', False):
            y = self.alex.predictor(image_array).data.argmax(axis=1)[0]
        labels = ["female", "male"]
        print(labels[y], flush=True)
        self.pub_result.publish(PredictResult(class_name=String(data=labels[y])))


def main():
    rclpy.init()
    node = GenderPredict("GenderPredictor")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
