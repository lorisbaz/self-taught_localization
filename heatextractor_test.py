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
        self.segm = ImgSegmFelzen(scales=[200, 300], sigmas=[0.4, 0.5], mins=[40])  
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
        print 'Heatmap computation using segmentation may take a while (around 30-40 seconds)...'
        heatmaps = self.heatext.extract(img,'n01751748')
        self.assertEqual(np.shape(heatmaps)[0], 4)
        self.assertAlmostEqual(np.sum(heatmaps[0].get_values()), 23.29836165, places=5)
        self.assertAlmostEqual(heatmaps[0].get_values()[50,50], 0.046209613, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[1].get_values()), 23.31647561, places=5)
        self.assertAlmostEqual(heatmaps[1].get_values()[70,70], 0.000112873, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[2].get_values()), 28.93356962, places=5)
        self.assertAlmostEqual(heatmaps[2].get_values()[50,50], 0.017993079, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[3].get_values()), 21.76335575, places=5)
        self.assertAlmostEqual(heatmaps[3].get_values()[50,50], 0.017954075, places=5)
        
        
#=============================================================================

class HeatmapExtractorBoxCaffe(unittest.TestCase):
    def setUp(self):
        self.conf = Configuration()
        self.net = NetworkCaffe(self.conf.ilsvrc2012_caffe_model_spec,\
                                self.conf.ilsvrc2012_caffe_model,\
                                self.conf.ilsvrc2012_caffe_wnids_words,\
                                self.conf.ilsvrc2012_caffe_avg_image)
        box_sz = [10, 20, 30]
        stride = 5
        self.heatext = HeatmapExtractorBox(self.net, box_sz, stride, \
                                           confidence_tech = 'only_obf')
        
    def tearDown(self):
        self.conf = None
        self.net = None
        self.heatext = None

    def test_extract_img(self):
        img = imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        img = resize(img, (100,100)) # resize fo faster computation
        print 'Heatmap computation using gray box may take a while (around 60-70 seconds)...'
        heatmaps = self.heatext.extract(img,'n01751748')
        self.assertEqual(np.shape(heatmaps)[0], 3)
        ## TODOOOOOOOOOOO ------
        #self.assertAlmostEqual(np.sum(heatmaps[0].get_values()), 332.3204620, places=5)
        #self.assertAlmostEqual(heatmaps[0].get_values()[50,50], 0.038591876, places=5)
        #self.assertAlmostEqual(np.sum(heatmaps[1].get_values()), 278.4503315, places=5)
        #self.assertAlmostEqual(heatmaps[1].get_values()[70,70], 0.038861134, places=5)
        #self.assertAlmostEqual(np.sum(heatmaps[2].get_values()), 220.6816997, places=5)
        #self.assertAlmostEqual(heatmaps[2].get_values()[50,50], 0.039973076, places=5)


#=============================================================================

# TODO here test slic segmentation, whenever it is implemented
# TODO here test DeCaf

#=============================================================================

if __name__ == '__main__':
    unittest.main()