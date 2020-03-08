import chainer
from chainer import links
from chainer import functions


class Alex(chainer.Chain):

    @classmethod
    def from_params(cls, *args, **kwargs):
        pass

    def __init__(self):
        super(Alex, self).__init__(
            conv1=links.Convolution2D(3, 48, 3, stride=1),
            bn1=links.BatchNormalization(48),
            conv2=links.Convolution2D(48, 128, 3, pad=1),
            bn2=links.BatchNormalization(128),
            conv3=links.Convolution2D(128, 192, 3, pad=1),
            conv4=links.Convolution2D(192, 192, 3, pad=1),
            conv5=links.Convolution2D(192, 128, 3, pad=1),
            fc6=links.Linear(None, 1024),
            fc7=links.Linear(None, 1024),
            fc8=links.Linear(None, 2)
        )

    def __call__(self, x):
        h = self.bn1(self.conv1(x))
        h = functions.max_pooling_2d(functions.relu(h), 3, stride=2)
        h = self.bn2(self.conv2(h))
        h = functions.max_pooling_2d(functions.relu(h), 3, stride=2)
        h = functions.relu(self.conv3(h))
        h = functions.relu(self.conv4(h))
        h = functions.relu(self.conv5(h))
        h = functions.max_pooling_2d(functions.relu(h), 2, stride=2)
        h = functions.dropout(functions.relu(self.fc6(h)))
        h = functions.dropout(functions.relu(self.fc7(h)))
        h = self.fc8(h)
        return h
