import os
from network import *
from configuration import *
import unittest
import numpy as np

#@unittest.skip("Skipping Decaf Test")
class NetworkDecafTest(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration()
        # deprecated way to construct the network
        self.net  = NetworkDecaf(self.conf.ilsvrc2012_decaf_model_spec,\
                                 self.conf.ilsvrc2012_decaf_model,\
                                 self.conf.ilsvrc2012_classid_wnid_words,\
                                 center_only = True)
        # using the factory
        params = NetworkDecafParams(self.conf.ilsvrc2012_decaf_model_spec,\
                                    self.conf.ilsvrc2012_decaf_model,\
                                    self.conf.ilsvrc2012_classid_wnid_words,\
                                    center_only = True)
        self.net2 = Network.create_network(params)

    def tearDown(self):
        self.net = None

    def test_get_input_dim(self):
        self.assertEqual(self.net.get_input_dim(), 227)

    def test_get_label_id(self):
        self.assertEqual(self.net.get_label_id('n04201297'), 0)
        self.assertEqual(self.net.get_label_id('n03063599'), 1)
        self.assertEqual(self.net.get_label_id('n03961711'), 999)

    def test_get_label_desc(self):
        self.assertEqual(self.net.get_label_desc('n04201297'), 'shoji')
        self.assertEqual(self.net.get_label_desc('n03063599'), 'coffee mug')
        self.assertEqual(self.net.get_label_desc('n03961711'), 'plate rack')

    def test_get_labels(self):
        labels = self.net.get_labels()
        self.assertEqual(labels[0], 'n04201297')
        self.assertEqual(labels[1], 'n03063599')
        self.assertEqual(labels[999], 'n03961711')

    def test_evaluate(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        scores = self.net.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(scores[0], 8.4301e-09, places=5)
        self.assertAlmostEqual(scores[999], 2.26509e-08, places=5)
        self.assertAlmostEqual(max(scores), 0.301472, places=5)
        scores = self.net.evaluate(img, layer_name = 'fc6_relu')
        self.assertEqual(scores.shape[0], 1)
        self.assertEqual(scores.shape[1], 4096)
        self.assertAlmostEqual(scores[0,0], 0.0)
        self.assertAlmostEqual(scores[0,3], 9.88611, places=4)
        self.assertAlmostEqual(scores[0,564], 3.55161, places=4)
        self.assertAlmostEqual(scores[0,3530], 15.6157, places=4)

    def test_evaluate2(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        scores = self.net2.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(scores[0], 8.4301e-09, places=5)
        self.assertAlmostEqual(scores[999], 2.26509e-08, places=5)
        self.assertAlmostEqual(max(scores), 0.301472, places=5)
        scores = self.net2.evaluate(img, layer_name = 'fc6_relu')
        self.assertEqual(scores.shape[0], 1)
        self.assertEqual(scores.shape[1], 4096)
        self.assertAlmostEqual(scores[0,0], 0.0)
        self.assertAlmostEqual(scores[0,3], 9.88611, places=4)
        self.assertAlmostEqual(scores[0,564], 3.55161, places=4)
        self.assertAlmostEqual(scores[0,3530], 15.6157, places=4)

#=============================================================================

class NetworkCaffeTest(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration()
        # deprecated way to construct the network
        self.net = NetworkCaffe(self.conf.ilsvrc2012_caffe_model_spec,\
                                self.conf.ilsvrc2012_caffe_model,\
                                self.conf.ilsvrc2012_caffe_wnids_words,\
                                self.conf.ilsvrc2012_caffe_avg_image, \
                                center_only = True)
        # using the factory
        params = NetworkCaffeParams(self.conf.ilsvrc2012_caffe_model_spec,\
                                    self.conf.ilsvrc2012_caffe_model,\
                                    self.conf.ilsvrc2012_caffe_wnids_words,\
                                    self.conf.ilsvrc2012_caffe_avg_image, \
                                    center_only = True)
        self.net2 = Network.create_network(params)
        # using the factory
        self.wnid_my_subset = ['n01440764', 'n01443537', 'n01751748']
        params2 = NetworkCaffeParams(self.conf.ilsvrc2012_caffe_model_spec,\
                                    self.conf.ilsvrc2012_caffe_model,\
                                    self.conf.ilsvrc2012_caffe_wnids_words,\
                                    self.conf.ilsvrc2012_caffe_avg_image, \
                                    center_only = True, \
                                    wnid_subset = self.wnid_my_subset)
        self.net3 = Network.create_network(params2)

    def tearDown(self):
        self.net = None

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

    def test_evaluate(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        scores = self.net.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(scores[0], 5.41017e-06, places=5)
        self.assertAlmostEqual(scores[999], 8.32369e-09, places=5)
        self.assertAlmostEqual(max(scores), 0.640054, places=5)

    def test_evaluate2(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        scores = self.net2.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(scores[0], 5.41017e-06, places=5)
        self.assertAlmostEqual(scores[999], 8.32369e-09, places=5)
        self.assertAlmostEqual(max(scores), 0.640054, places=5)

    def test_evaluate_subset(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        scores = self.net3.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(\
            scores[self.net3.get_label_id(self.wnid_my_subset[0])], \
            5.4101706e-06, places=6)
        self.assertAlmostEqual(\
            scores[self.net3.get_label_id(self.wnid_my_subset[1])], \
            4.0534101e-08, places=7)
        self.assertAlmostEqual(\
            scores[self.net3.get_label_id(self.wnid_my_subset[2])], \
            0.026647, places=5)
        self.assertAlmostEqual(\
            scores[self.net3.get_label_id('n02892767')], 0.0, places=5)

    def test_evaluate_layers(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        features = self.net.evaluate(img, layer_name = 'fc7')
        self.assertAlmostEqual(features[0], 0.0, places=5)
        self.assertAlmostEqual(features[4095], 0.0, places=5)
        self.assertAlmostEqual(max(features), 10.250695, places=5)
        features = self.net.evaluate(img, layer_name = 'pool5')
        self.assertAlmostEqual(features[0,0,0], 4.2559791, places=5)
        self.assertAlmostEqual(features[255,5,5], 0.0, places=5)
        self.assertAlmostEqual(np.max(features), 77.13868, places=5)
        features = self.net.evaluate(img, layer_name = 'conv2')
        self.assertAlmostEqual(features[0,0,0], 0.0, places=5)
        self.assertAlmostEqual(features[255,10,10], 0.0, places=5)
        self.assertAlmostEqual(np.max(features), 335.69110, places=5)

    def test_evaluate_layers2(self):
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        features = self.net2.evaluate(img, layer_name = 'fc7')
        self.assertAlmostEqual(features[0], 0.0, places=5)
        self.assertAlmostEqual(features[4095], 0.0, places=5)
        self.assertAlmostEqual(max(features), 10.250695, places=5)
        features = self.net2.evaluate(img, layer_name = 'pool5')
        self.assertAlmostEqual(features[0,0,0], 4.2559791, places=5)
        self.assertAlmostEqual(features[255,5,5], 0.0, places=5)
        self.assertAlmostEqual(np.max(features), 77.13868, places=5)
        features = self.net2.evaluate(img, layer_name = 'conv2')
        self.assertAlmostEqual(features[0,0,0], 0.0, places=5)
        self.assertAlmostEqual(features[255,10,10], 0.0, places=5)
        self.assertAlmostEqual(np.max(features), 335.69110, places=5)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
