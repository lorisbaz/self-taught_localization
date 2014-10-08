import numpy as np

from util import *
from network import *

class ReRanker:

    def __init__(self):
        pass

    def evaluate(self, bbox):
        pass


class ReRankerNet(ReRanker):

    def __init__(self, network, layer='softmax'):

        self.network_ = network
        self.layer_ = layer

    def evaluate(self, image, bb):
        """
        Given a bbox, it will rerank it with a new network. We take the max of
        the softmax output.
        """
        # get the top1 label
        top1 = np.argmax(self.network_.evaluate(image, \
                                                layer_name = self.layer_))
        image_height, image_width = np.shape(image)[0:2]
        # rescale
        bbox = bb.copy()
        bbox.rescale_to_outer_box(image_width, image_height)
        bbox.convert_coordinates_to_integers()
        # crop image
        image_box = np.copy(image[bbox.ymin:bbox.ymax, bbox.xmin:bbox.xmax])
        # predict CNN reponse for obfuscation
        caffe_rep = self.network_.evaluate(image_box, layer_name = self.layer_)
        # take the max
        return caffe_rep[top1]
