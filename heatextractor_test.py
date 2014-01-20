import numpy as np
from skimage.data import imread
import unittest
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *

class HeatmapExtractorSegmCaffe(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration()
        self.net = NetworkCaffe(self.conf.ilsvrc2012_caffe_model_spec,\
                                self.conf.ilsvrc2012_caffe_model,\
                                self.conf.ilsvrc2012_caffe_wnids_words,\
                                self.conf.ilsvrc2012_caffe_avg_image)
        self.segm = ImgSegmFelzen(sigmas=[0.1, 0.4], mins=[40], scales=[100, 300])  
        self.heatext = HeatmapExtractorSegm(self.net, self.segm)
        
    def tearDown(self):
        self.conf = None
        self.net = None
        self.segm = None   
        self.heatext = None

    def test_extract(self):
        img = imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        heatmaps = self.heatext.extract(img,'n01751748')
        self.assertEqual(heatmaps, [])

#=============================================================================

# TODO here test slic segmentation, whenever it is implemented
# TODO here test DeCaf

#=============================================================================

if __name__ == '__main__':
    unittest.main()