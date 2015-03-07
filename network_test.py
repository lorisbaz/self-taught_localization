import os
from network import *
from configuration import *
import unittest
import numpy as np

class NetworkCaffe1114Test(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration(caffe_model = 'alexnet')
        # deprecated way to construct the network
        self.net = NetworkCaffe1114(self.conf.ilsvrc2012_caffe_model_spec,\
                                self.conf.ilsvrc2012_caffe_model,\
                                self.conf.ilsvrc2012_caffe_wnids_words,\
                                self.conf.ilsvrc2012_caffe_avg_image, \
                                center_only = True)
        # using the factory
        params = NetworkCaffe1114Params(self.conf.ilsvrc2012_caffe_model_spec,\
                                    self.conf.ilsvrc2012_caffe_model,\
                                    self.conf.ilsvrc2012_caffe_wnids_words,\
                                    self.conf.ilsvrc2012_caffe_avg_image, \
                                    center_only = True)
        self.net2 = Network.create_network(params)
        # using the factory
        self.wnid_my_subset = ['n01440764', 'n01443537', 'n01751748']
        params2 = NetworkCaffe1114Params(self.conf.ilsvrc2012_caffe_model_spec,\
                                    self.conf.ilsvrc2012_caffe_model,\
                                    self.conf.ilsvrc2012_caffe_wnids_words,\
                                    self.conf.ilsvrc2012_caffe_avg_image, \
                                    center_only = True, \
                                    wnid_subset = self.wnid_my_subset)
        self.net3 = Network.create_network(params2)

    def tearDown(self):
        self.net = None
        self.net2 = None
        self.net3 = None

    def test_get_label_id(self):
        self.assertEqual(self.net.get_label_id('n01440764'), 0)
        self.assertEqual(self.net.get_label_id('n01443537'), 1)
        self.assertEqual(self.net.get_label_id('n15075141'), 999)

    def test_get_label_desc(self):
        self.assertEqual(self.net.get_label_desc('n01440764'), \
                         'tench, Tinca tinca')
        self.assertEqual(self.net.get_label_desc('n01443537'), \
                         'goldfish, Carassius auratus')
        self.assertEqual(self.net.get_label_desc('n15075141'), \
                         'toilet tissue, toilet paper, bathroom tissue')

    def test_get_labels(self):
        labels = self.net.get_labels()
        self.assertEqual(labels[0], 'n01440764')
        self.assertEqual(labels[1], 'n01443537')
        self.assertEqual(labels[999], 'n15075141')

    def test_evaluate2(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        scores = self.net2.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(scores[0], 5.4873649e-06, places=5)
        self.assertAlmostEqual(scores[999], 7.0695227e-09, places=5)
        self.assertAlmostEqual(max(scores), 0.28026706, places=5)

    def test_evaluate_layers2(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        features = self.net2.evaluate(img, layer_name = 'fc7')
        self.assertAlmostEqual(features[0], 0.0, places=5)
        self.assertAlmostEqual(features[4095], 0.0, places=5)
        self.assertAlmostEqual(max(features), 11.615518, places=5)
        features = self.net2.evaluate(img, layer_name = 'pool5')
        self.assertAlmostEqual(features[0,0,0], 0.0, places=5)
        self.assertAlmostEqual(features[255,5,5], 0.0, places=5)
        self.assertAlmostEqual(np.max(features), 77.559799, places=5)
        features = self.net2.evaluate(img, layer_name = 'conv2')
        self.assertAlmostEqual(features[0,0,0], 0.0, places=5)
        self.assertAlmostEqual(features[100,10,10], 54.716732, places=5)
        self.assertAlmostEqual(np.max(features), 276.84589, places=4)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
