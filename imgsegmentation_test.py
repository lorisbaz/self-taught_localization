import numpy as np
from skimage.data import imread
import unittest

class ImgSegmFelzen(unittest.TestCase):
    def setUp(self):
        from imgsegmentation import *     
        self.segmenter1 = ImgSegmFelzen(sigmas=[0.4], mins=[40], scales=[300]) # test single segmentations
        self.segmenter2 = ImgSegmFelzen(sigmas=[0.1, 0.4], mins=[40], scales=[100, 300]) # test multiple segmentations
        
    def tearDown(self):
        self.segmenter1 = None
        self.segmenter2 = None

    def test_extract(self):
        img = imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        segm_mask = self.segmenter1.extract(img)
        self.assertEqual(segm_mask[0][100,10], 131)
        self.assertEqual(segm_mask[0][100,100], 268)
        self.assertEqual(np.max(segm_mask[0]), 671)
        segm_masks = self.segmenter2.extract(img)
        self.assertEqual(segm_masks[0][100,10], 282)
        self.assertEqual(segm_masks[0][100,100], 32)
        self.assertEqual(np.max(segm_masks[0]), 1669)
        self.assertEqual(segm_masks[2][100,10], 126)
        self.assertEqual(segm_masks[2][100,100], 310)
        self.assertEqual(np.max(segm_masks[2]), 1642)
        self.assertEqual(segm_masks[3][100,10], 131)
        self.assertEqual(segm_masks[3][100,100], 268)
        self.assertEqual(np.max(segm_masks[3]), 671)

#=============================================================================

# TODO here test slic segmentation, whenever it is implemented

#=============================================================================

if __name__ == '__main__':
    unittest.main()
