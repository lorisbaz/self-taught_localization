import numpy as np
from skimage.data import imread
import unittest

class ImgSegmFelzen(unittest.TestCase):
    def setUp(self):
        # TODO fix this
        from imgsegmentation import *
        # test single segmentations
        self.segmenter1 = ImgSegmFelzen(scales = [300], sigmas = [0.4], \
                                        min_sizes = [40]) 
        # test multiple segmentations
        self.segmenter2 = ImgSegmFelzen(scales = [200, 300], \
                                        sigmas = [0.3, 0.4], \
                                        min_sizes = [40]) 
        
    def tearDown(self):
        self.segmenter1 = None
        self.segmenter2 = None

    def test_extract(self):
        img = imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        segm_mask = self.segmenter1.extract(img)
        self.assertEqual(segm_mask[0][100,10], 127)
        self.assertEqual(segm_mask[0][100,100], 0)
        self.assertEqual(np.max(segm_mask[0]), 662)
        segm_masks = self.segmenter2.extract(img)
        self.assertEqual(segm_masks[0][100,10], 0)
        self.assertEqual(segm_masks[0][100,100], 0)
        self.assertEqual(np.max(segm_masks[0]), 0)
        self.assertEqual(segm_masks[2][100,10], 66)
        self.assertEqual(segm_masks[2][100,100], 0)
        self.assertEqual(np.max(segm_masks[2]), 409)
        self.assertEqual(segm_masks[3][100,10], 127)
        self.assertEqual(segm_masks[3][100,100], 0)
        self.assertEqual(np.max(segm_masks[3]), 662)

#=============================================================================

# TODO here test slic segmentation, whenever it is implemented

#=============================================================================

if __name__ == '__main__':
    unittest.main()
