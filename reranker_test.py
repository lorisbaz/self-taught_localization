import numpy as np
from skimage.data import imread
import unittest

from bbox import *
from configuration import *
from network import *
from reranker import *

class ReRankerTest(unittest.TestCase):
    def setUp(self):
        conf = Configuration()
        netParams = NetworkCaffeParams(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = True,\
                           wnid_subset = [])
        # instantiate network
        self.net = Network.create_network(netParams)
        # instanciate reranker
        self.reranker = ReRankerNet(self.net)
        assert isinstance(self.reranker, ReRanker)
        # bbox
        self.bboxes = [BBox(0,0,0.5,0.5,1.0),\
                        BBox(0.2,0.2,0.6,0.6,1.0),\
                        BBox(0.5,0.5,0.5,1.0,1.0)]

    def tearDown(self):
        self.net = None
        self.reranker = None
        self.bboxes = None

    def test_reranker(self):
        image = skimage.io.imread('test_data/'+\
                            'ILSVRC2012_val_00000001_n01751748.JPEG')
        for bb in self.bboxes:
            bb.confidence = self.reranker.evaluate(image, bb)
        self.assertAlmostEqual(self.bboxes[0].confidence, 0.01585524)
        self.assertAlmostEqual(self.bboxes[1].confidence, 0.08206509)
        self.assertAlmostEqual(self.bboxes[2].confidence, 0.00354953)

# ------------ perform test ------------ #
if __name__ == '__main__':
    unittest.main()
