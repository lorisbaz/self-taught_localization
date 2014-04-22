import cv2
import numpy as np
import unittest

import bboxextractor

class BBoxExtractorTest(unittest.TestCase):
    def setUp(self):
        self.mask = np.array(   [[0,   1,   0,   0,   2], \
                                 [1,   2,   2,   0,   1]], np.int32)
        self.heatmap = np.array([[0.1, 0.2, 0.3, 0.4, 0.5], \
                                 [0.6, 0.7, 0.8, 0.9, 1.0]], np.float64)

    def tearDown(self):
        self.mask = None
        self.heatmap = None

    def test_get_bbox_from_connected_components(self):
        object_values = [1, 2]
        bboxes = \
            bboxextractor.BBoxExtractor.get_bbox_from_connected_components_( \
              self.mask, self.heatmap, object_values)
        self.assertEqual(isinstance(bboxes, list), True)
        self.assertEqual(len(bboxes), 2)
        self.assertEqual(bboxes[0].xmin, 0)
        self.assertEqual(bboxes[0].ymin, 0)
        self.assertEqual(bboxes[0].xmax, 3)
        self.assertEqual(bboxes[0].ymax, 2)
        self.assertEqual(bboxes[0].confidence, (0.1+0.2+0.3+0.6+0.7+0.8)/6.0)
        self.assertEqual(bboxes[1].xmin, 4)
        self.assertEqual(bboxes[1].ymin, 0)
        self.assertEqual(bboxes[1].xmax, 5)
        self.assertEqual(bboxes[1].ymax, 2)
        self.assertEqual(bboxes[1].confidence, (0.5+1.0)/2.0)

#=============================================================================

class GrabCutBBoxExtractor(unittest.TestCase):
    def setUp(self):
        self.extract = bboxextractor.GrabCutBBoxExtractor()

    def tearDown(self):
        self.extract = None

    def test_get_thresholds_from_kmeans(self):
        heatmap = np.array([[0.1, 0.1, 0.1, 0.5, 0.5, 0.5], \
                            [1.0, 1.0, 1.0, 1.5, 1.5, 1.5]], np.float64)
        thresholds = self.extract.get_thresholds_from_kmeans_(heatmap)
        self.assertAlmostEqual(thresholds[0], 0.3)
        self.assertAlmostEqual(thresholds[1], 0.75)
        self.assertAlmostEqual(thresholds[2], 1.25)

    def test_get_grabcut_init_mask_(self):
        heatmap = np.array([[0.7, 0.2, 0.1, 0.8], \
                            [0.6, 0.3, 0.4, 0.9]], np.float64)
        thresholds = [0.3, 0.5, 0.7]
        mask = self.extract.get_grabcut_init_mask_(heatmap, thresholds)
        mask_excepted = np.array( \
              [[cv2.GC_FGD, cv2.GC_BGD, cv2.GC_BGD, cv2.GC_FGD], \
               [cv2.GC_PR_FGD, cv2.GC_PR_BGD, cv2.GC_PR_BGD, cv2.GC_FGD]])
        np.testing.assert_equal(mask, mask_excepted)

    def test_extract(self):
        test_image = 'test_data/ILSVRC2012_val_00000001_n01751748.JPEG'
        test_mask = 'test_data/ILSVRC2012_val_00000001_n01751748_test_mask.png'
        mask = cv2.imread(test_mask)
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        img = cv2.imread(test_image)
        bboxes, image_desc = self.extract.extract(img, [mask])
        idx = 0
        for img, desc in image_desc:
            print desc
            cv2.imwrite('imgdesc{0}.png'.format(idx), img)
            idx += 1

#=============================================================================

if __name__ == '__main__':
    unittest.main()
