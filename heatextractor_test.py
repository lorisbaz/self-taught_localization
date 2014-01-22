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
        self.segm = ImgSegmFelzen(sigmas=[0.4, 0.6], mins=[40], scales=[100, 300])  
        self.heatext = HeatmapExtractorSegm(self.net, self.segm, confidence_tech = 'only_obf')
        
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
        self.assertAlmostEqual(np.sum(heatmaps[0].get_values()), 7.226420715, places=5)
        self.assertAlmostEqual(heatmaps[0].get_values()[50,50], 0.001178693, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[1].get_values()), 0.421182818, places=5)
        self.assertAlmostEqual(heatmaps[1].get_values()[70,70], 0.0, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[2].get_values()), 6.376812301, places=5)
        self.assertAlmostEqual(heatmaps[2].get_values()[50,50], 0.001235786, places=5)
        self.assertAlmostEqual(np.sum(heatmaps[3].get_values()), 0.302846163, places=5)
        self.assertAlmostEqual(heatmaps[3].get_values()[50,50], 8.763283776e-05, places=5)
        
#=============================================================================

# TODO here test slic segmentation, whenever it is implemented
# TODO here test DeCaf

#=============================================================================

if __name__ == '__main__':
    unittest.main()