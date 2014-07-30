import numpy as np
from skimage.data import imread
import unittest

from imgsegmentation import *
from configuration import *
from network import *
from self_taught_loc import *

class SelfTaughtLocTest(unittest.TestCase):
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
        # choose segmentation method (Matlab wrapper Felz through SS)
        img_segmenter = ImgSegmMatWraper()
        # instantiate STL object
        self.stl_grayout = SelfTaughtLoc_Grayout(self.net, img_segmenter, \
                                   min_sz_segm = 5, topC = 5,\
                                   alpha = 1/3.0*np.ones((3,)), \
                                   obfuscate_bbox = True, \
                                   function_stl = 'similarity')
        # instantiate STL object
        self.stl_grayout_2 = SelfTaughtLoc_Grayout(self.net, img_segmenter, \
                                    min_sz_segm = 5, topC = 5,\
                                    alpha = 1/4.0*np.ones((4,)), \
                                    obfuscate_bbox = True, \
                                    function_stl = 'similarity+cnnfeature')
        # instantiate STL object
        self.stl_grayout_3 = SelfTaughtLoc_Grayout(self.net, img_segmenter, \
                                   min_sz_segm = 5, topC = 5,
                                   alpha = 1/4.0*np.ones((4,)), \
                                   obfuscate_bbox = True, \
                                   function_stl = 'similarity+cnnfeature', \
                                   padding = 0.2)
    def tearDown(self):
        self.stl_grayout = None
        self.stl_grayout_2 = None
        self.stl_grayout_3 = None

    def test_extract_stl_u(self):
        # read image
        img = imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        # resize image to fit the net input
        image_resz = skimage.transform.resize(img,\
                            (self.net.get_input_dim(), self.net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz)
        img_width, img_height = np.shape(image_resz)[0:2]
        # perform stl unsupervised
        segment_lists = self.stl_grayout.extract_greedy(image_resz)
        # Control some elements
        self.assertEqual(np.shape(segment_lists)[0], 4)
        self.assertEqual(np.shape(segment_lists[0])[0], 217)
        self.assertEqual(np.shape(segment_lists[1])[0], 123)
        self.assertEqual(np.shape(segment_lists[2])[0], 93)
        self.assertEqual(np.shape(segment_lists[3])[0], 47)
        self.assertEqual(segment_lists[0][100]['mask'][0,0], False)
        self.assertEqual(segment_lists[1][50]['mask'][10,10], True)
        self.assertEqual(segment_lists[2][65]['mask'][0,0], False)
        self.assertEqual(segment_lists[3][33]['mask'][30,10], True)
        self.assertEqual(segment_lists[0][100]['bbox'].xmin, 87)
        self.assertEqual(segment_lists[1][50]['bbox'].xmin, 41)
        self.assertEqual(segment_lists[2][65]['bbox'].xmin, 9)
        self.assertEqual(segment_lists[3][33]['bbox'].xmin, 73)

    def test_extract_stl_u_cnnfeature(self):
        # read image
        img = imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        # resize image to fit the net input
        image_resz = skimage.transform.resize(img,\
                            (self.net.get_input_dim(), self.net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz)
        img_width, img_height = np.shape(image_resz)[0:2]
        # perform stl unsupervised
        segment_lists = self.stl_grayout_2.extract_greedy(image_resz)
        # Control some elements
        self.assertEqual(np.shape(segment_lists)[0], 4)
        self.assertEqual(np.shape(segment_lists[0])[0], 217)
        self.assertEqual(np.shape(segment_lists[1])[0], 123)
        self.assertEqual(np.shape(segment_lists[2])[0], 93)
        self.assertEqual(np.shape(segment_lists[3])[0], 47)
        self.assertEqual(segment_lists[0][100]['mask'][0,0], False)
        self.assertEqual(segment_lists[1][50]['mask'][10,10], True)
        self.assertEqual(segment_lists[2][65]['mask'][0,0], False)
        self.assertEqual(segment_lists[3][33]['mask'][30,10], False)
        self.assertEqual(segment_lists[0][100]['bbox'].xmin, 87)
        self.assertEqual(segment_lists[1][50]['bbox'].xmin, 41)
        self.assertEqual(segment_lists[2][65]['bbox'].xmin, 170)
        self.assertEqual(segment_lists[3][33]['bbox'].xmin, 0)

    def test_extract_stl_u_cnnfeature_pad(self):
        # read image
        img = imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        # resize image to fit the net input
        image_resz = skimage.transform.resize(img,\
                            (self.net.get_input_dim(), self.net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz)
        img_width, img_height = np.shape(image_resz)[0:2]
        # perform stl unsupervised
        segment_lists = self.stl_grayout_3.extract_greedy(image_resz)
        # Control some elements
        self.assertEqual(np.shape(segment_lists)[0], 4)
        self.assertEqual(np.shape(segment_lists[0])[0], 217)
        self.assertEqual(np.shape(segment_lists[1])[0], 123)
        self.assertEqual(np.shape(segment_lists[2])[0], 93)
        self.assertEqual(np.shape(segment_lists[3])[0], 47)
        self.assertEqual(segment_lists[0][100]['mask'][0,0], False)
        self.assertEqual(segment_lists[1][50]['mask'][10,10], True)
        self.assertEqual(segment_lists[2][65]['mask'][0,0], False)
        self.assertEqual(segment_lists[3][33]['mask'][30,10], False)
        self.assertEqual(segment_lists[0][100]['bbox'].xmin, 87)
        self.assertEqual(segment_lists[1][50]['bbox'].xmin, 41)
        self.assertEqual(segment_lists[2][65]['bbox'].xmin, 82)
        self.assertEqual(segment_lists[3][33]['bbox'].xmin, 47)


# ------------ perform test ------------ #
if __name__ == '__main__':
    unittest.main()
