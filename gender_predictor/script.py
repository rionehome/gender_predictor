import os

import rclpy
from rclpy.node import Node
import numpy as np
from keras import models

from rione_msgs.msg import PredictResult
from sensor_msgs.msg import Image
from std_msgs.msg import String

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../gender_predictor/models/")


class GenderPredict(Node):
    def __init__(self, node_name: str):
        super().__init__(node_name)
        self.create_subscription(Image, "/gender_predictor/color/image", self.callback_image, 1)
        self.pub_result = self.create_publisher(PredictResult, "/gender_predictor/result", 10)
        self.model = models.load_model("{}/gender_vgg16.model".format(MODEL_PATH))
        print("load complete!", flush=True)

    def callback_image(self, msg: Image):
        """
        画像のsubscribe
        :param msg:
        :return:
        """
        if not len(msg.data) == 96 ** 2 * 3:
            print("画像サイズは96×96×3である必要があります", flush=True)
            return
        image_array = np.asarray(msg.data).reshape((1, 96, 96, 3)).astype(np.float32) / 255.
        y = self.model.predict(image_array).argmax(axis=1)[0]
        labels = ["female", "male"]
        self.pub_result.publish(PredictResult(class_name=String(data=labels[y])))


def main():
    rclpy.init()
    node = GenderPredict("GenderPredictor")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
