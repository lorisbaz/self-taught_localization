import os
from network import *
from configuration import *
import unittest


class NetworkDecafTest(unittest.TestCase):
    def setUp(self):
	self.conf = Configuration()
        self.net = NetworkDecaf(self.conf.ilsvrc2012_decaf_model_spec,\
				self.conf.ilsvrc2012_decaf_model,\
				self.conf.ilsvrc2012_classid_wnid_words)

    def tearDown(self):
        self.net = None

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
	img = np.asarray(io.imread('ILSVRC2012_val_00000001_n01751748.JPEG'))
	scores = self.net.evaluate(img, layer_name = 'softmax')
        self.assertAlmostEqual(scores[0], 6.9254136e-09)
	self.assertAlmostEqual(scores[999], 1.2282071e-08)
	self.assertAlmostEqual(max(scores), 0.36385602)
	scores = self.net.evaluate(img, layer_name = 'fc6_relu')
	self.assertEqual(scores.shape[0], 1)
	self.assertEqual(scores.shape[1], 4096)
        self.assertAlmostEqual(scores[0,0], 0.0)
	self.assertAlmostEqual(scores[0,3], 6.447751)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
