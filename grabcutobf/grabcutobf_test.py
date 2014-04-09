import cv2
import numpy as np
import unittest

import grabcutobf

class GrabCutObfTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_grabcutobfs(self):
        img = cv2.imread("airplane.png")
        rect = (24, 126, 483, 294)
        # original OpenCV GrabCut
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        mask = np.zeros(img.shape[0:2], np.uint8)
        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, \
                    5, cv2.GC_INIT_WITH_RECT)
        mask1 = mask.copy()
        cv2.imshow('mask1', mask1*40)
        cv2.waitKey(0)
        # our modified GrabCut
        bgdModel = np.zeros((1,65), np.float64)
        fgdModel = np.zeros((1,65), np.float64)
        mask = np.zeros(img.shape[0:2], np.uint8)
        grabcutobf.grabCutObf(img, mask, rect, bgdModel, fgdModel, \
                              5, cv2.GC_INIT_WITH_RECT)
        mask2 = mask.copy()
        cv2.imshow('mask2', mask2*40)
        cv2.waitKey(0)
        #np.testing.assert_allclose(mask1, mask2)

#=============================================================================

if __name__ == '__main__':
    unittest.main()

