import numpy as np
from skimage.data import imread
from skimage.transform import resize
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
        self.segm = ImgSegmFelzen(scales=[200, 300], sigmas=[0.4, 0.5], min_sizes=[40])  
        self.heatext = HeatmapExtractorSegm(self.net, self.segm, \
                                            confidence_tech = 'only_obf')
        
    def tearDown(self):
        self.conf = None
        self.net = None
        self.segm = None   
        self.heatext = None

    def test_extract_img(self):
        img = imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        img = resize(img, (100,100)) # resize fo faster computation
        img = skimage.img_as_ubyte(img)
        print 'Heatmap computation using segmentation may take a while (around 30-40 seconds)...'
        heatmaps = self.heatext.extract(img,'n01751748')
        self.assertEqual(np.shape(heatmaps)[0], 4)
        self.assertAlmostEqual(np.sum(heatmaps[0].get_values()), 35.78203093, places=5)
        self.assertAlmostEqual(heatmaps[0].get_values()[50,50], 0.018860068, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[1].get_values()), 20.15715549, places=5)
        self.assertAlmostEqual(heatmaps[1].get_values()[70,70], 0.000115786, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[2].get_values()), 31.30557203, places=5)
        self.assertAlmostEqual(heatmaps[2].get_values()[50,50], 0.017164974, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[3].get_values()), 20.06403474, places=5)
        self.assertAlmostEqual(heatmaps[3].get_values()[50,50], 0.016967707, places=5)
        
        
#=============================================================================

class HeatmapExtractorBoxCaffe(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration()
        self.net = NetworkCaffe(self.conf.ilsvrc2012_caffe_model_spec,\
                                self.conf.ilsvrc2012_caffe_model,\
                                self.conf.ilsvrc2012_caffe_wnids_words,\
                                self.conf.ilsvrc2012_caffe_avg_image)
        params = [(10, 10), \
		  (30, 10)]
        self.heatext = HeatmapExtractorBox(self.net, params, \
                                           confidence_tech = 'only_obf')
         
    def tearDown(self):
        self.conf = None
        self.net = None
        self.heatext = None

    def test_extract_img(self):
        img = imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        img = resize(img, (100,100)) # resize fo faster computation
        img = skimage.img_as_ubyte(img)
        print 'Heatmap computation using gray box may take a while (around 60-70 seconds)...'
        heatmaps = self.heatext.extract(img,'n01751748')
        self.assertEqual(np.shape(heatmaps)[0], 2)
        self.assertAlmostEqual(np.sum(heatmaps[0].get_values()), 89.78405883, places=5)
        self.assertAlmostEqual(heatmaps[0].get_values()[50,50], 0.009513479, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[1].get_values()), 10.88570486, places=5)
        self.assertAlmostEqual(heatmaps[1].get_values()[70,70], 0.001089485, places=5)
        
        
#=============================================================================

class HeatmapExtractorSlidingCaffe(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration()
        self.net = NetworkCaffe(self.conf.ilsvrc2012_caffe_model_spec,\
                                self.conf.ilsvrc2012_caffe_model,\
                                self.conf.ilsvrc2012_caffe_wnids_words,\
                                self.conf.ilsvrc2012_caffe_avg_image)
        params = [(20, 10), \
		  (40, 10)]
        self.heatext = HeatmapExtractorSliding(self.net, params, \
                                               confidence_tech = 'only_win')
        
    def tearDown(self):
        self.conf = None
        self.net = None
        self.heatext = None

    def test_extract_img(self):
        img = imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        img = resize(img, (100,100)) # resize fo faster computation
        img = skimage.img_as_ubyte(img)
        print 'Heatmap computation using gray box may take a while (around 60-70 seconds)...'
        heatmaps = self.heatext.extract(img,'n01751748')
        self.assertEqual(np.shape(heatmaps)[0], 2)
        self.assertAlmostEqual(np.sum(heatmaps[0].get_values()), 24.98861628, places=5)
        self.assertAlmostEqual(heatmaps[0].get_values()[50,50], 0.002499311, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[1].get_values()), 6.227268339, places=5)
        self.assertAlmostEqual(heatmaps[1].get_values()[70,70], 0.000624565, places=5)
        
#=============================================================================

# TODO here test DeCaf

#=============================================================================

if __name__ == '__main__':
    unittest.main()
