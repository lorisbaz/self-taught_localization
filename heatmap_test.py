import os
from heatmap import *
import numpy as np
import tempfile
import unittest

class NetworkDecafTest(unittest.TestCase):
    def setUp(self):
        self.heat = Heatmap(3, 2)
        self.heat2 = Heatmap(300, 200)

    def tearDown(self):
        self.heat = None
        self.heat2 = None

    def test_init(self):
        self.assertEqual(self.heat.vals_[0,0], 0.0)
        self.assertEqual(self.heat.vals_[1,2], 0.0)
        self.assertEqual(self.heat.counts_[0,0], 0)
        self.assertEqual(self.heat.counts_[1,2], 0)

    def test_add_val_rect1(self):
        self.heat.add_val_rect(0.1, 1, 1, width=2, height=1, \
                               area_normalization = True)
        self.assertEqual(self.heat.vals_[1,0], 0.0)
        self.assertEqual(self.heat.vals_[1,1], 0.05)
        self.assertEqual(self.heat.vals_[1,2], 0.05)
        self.assertEqual(self.heat.counts_[1,0], 0)
        self.assertEqual(self.heat.counts_[1,1], 1)
        self.assertEqual(self.heat.counts_[1,2], 1)
        self.heat.add_val_rect(0.3, 1, 1, width=1, height=1, \
                               area_normalization = True)
        self.assertEqual(self.heat.vals_[1,0], 0.0)
        self.assertEqual(self.heat.vals_[1,1], 0.05+0.3)
        self.assertEqual(self.heat.vals_[1,2], 0.05)
        self.assertEqual(self.heat.counts_[1,0], 0)
        self.assertEqual(self.heat.counts_[1,1], 2)
        self.assertEqual(self.heat.counts_[1,2], 1)

    def test_add_val_rect2(self):
        self.heat.add_val_rect(0.1, 1, 1, width=2, height=1, \
                               area_normalization = False)
        self.assertEqual(self.heat.vals_[1,0], 0.0)
        self.assertEqual(self.heat.vals_[1,1], 0.1)
        self.assertEqual(self.heat.vals_[1,2], 0.1)
        self.assertEqual(self.heat.counts_[1,0], 0)
        self.assertEqual(self.heat.counts_[1,1], 1)
        self.assertEqual(self.heat.counts_[1,2], 1)
        self.heat.add_val_rect(0.3, 1, 1, width=1, height=1, \
                               area_normalization = False)
        self.assertEqual(self.heat.vals_[1,0], 0.0)
        self.assertEqual(self.heat.vals_[1,1], 0.1+0.3)
        self.assertEqual(self.heat.vals_[1,2], 0.1)
        self.assertEqual(self.heat.counts_[1,0], 0)
        self.assertEqual(self.heat.counts_[1,1], 2)
        self.assertEqual(self.heat.counts_[1,2], 1)

    def test_add_val_segment1(self):
        seg_map = np.array([[0, 1, 0], [2, 1, 2]], np.int32)
        self.heat.add_val_segment(0.1, 0, seg_map, True)
        self.assertEqual(self.heat.vals_[0,0], 0.05)
        self.assertEqual(self.heat.vals_[0,1], 0.0)
        self.assertEqual(self.heat.vals_[0,2], 0.05)
        self.assertEqual(self.heat.counts_[0,0], 1)
        self.assertEqual(self.heat.counts_[0,1], 0)
        self.assertEqual(self.heat.counts_[0,2], 1)
        seg_map = np.array([[1, 2, 3], [1, 2, 3]], np.int32)
        self.heat.add_val_segment(0.1, 1, seg_map, True)
        self.assertEqual(self.heat.vals_[0,0], 0.05+0.05)
        self.assertEqual(self.heat.vals_[0,1], 0.0)
        self.assertEqual(self.heat.vals_[0,2], 0.05)
        self.assertEqual(self.heat.counts_[0,0], 2)
        self.assertEqual(self.heat.counts_[0,1], 0)
        self.assertEqual(self.heat.counts_[0,2], 1)

    def test_add_val_segment2(self):
        seg_map = np.array([[0, 1, 0], [2, 1, 2]], np.int32)
        self.heat.add_val_segment(0.1, 0, seg_map, False)
        self.assertEqual(self.heat.vals_[0,0], 0.1)
        self.assertEqual(self.heat.vals_[0,1], 0.0)
        self.assertEqual(self.heat.vals_[0,2], 0.1)
        self.assertEqual(self.heat.counts_[0,0], 1)
        self.assertEqual(self.heat.counts_[0,1], 0)
        self.assertEqual(self.heat.counts_[0,2], 1)
        seg_map = np.array([[1, 2, 3], [1, 2, 3]], np.int32)
        self.heat.add_val_segment(0.3, 1, seg_map, False)
        self.assertEqual(self.heat.vals_[0,0], 0.1+0.3)
        self.assertEqual(self.heat.vals_[0,1], 0.0)
        self.assertEqual(self.heat.vals_[0,2], 0.1)
        self.assertEqual(self.heat.counts_[0,0], 2)
        self.assertEqual(self.heat.counts_[0,1], 0)
        self.assertEqual(self.heat.counts_[0,2], 1)

    def test_normalize_counts(self):
        self.heat.vals_[0,0] = 3.0
        self.heat.counts_[0,0] = 2;
        self.heat.normalize_counts()
        self.assertEqual(self.heat.vals_[0,0], 3.0/2.0)
        self.assertEqual(self.heat.vals_[0,1], 0.0)
        self.assertEqual(self.heat.counts_[0,0], 1)
        self.assertEqual(self.heat.counts_[0,1], 1)

    def test_get_values(self):
        self.heat.vals_[0,0] = 3.0
        self.heat.counts_[0,0] = 2;
        self.assertEqual(self.heat.get_values()[0,0], 3.0)
        self.assertEqual(self.heat.get_values()[0,1], 0.0)

    def test_export_and_save_to_jpeg(self):
        """
        This test generates an an image with a gray gradient from
        left (black) to right (white).
        TODO. Test that the gradient saved in the jpeg file does actually
              match the expected one.
        """
        for y in range(self.heat2.vals_.shape[0]):
            for x in range(self.heat2.vals_.shape[1]):
                self.heat2.vals_[y,x] = x / (1.5 * self.heat2.vals_.shape[1])
                self.heat2.counts_[y,x] = 1
        image = self.heat2.export_to_image()
        (fd, filename) = tempfile.mkstemp(suffix = '.jpg')
        os.close(fd)
        self.heat2.save_to_image(filename)
        os.remove(filename)

    def test_sum_heatmaps(self):
        heat1 = Heatmap(2, 3)
        heat2 = Heatmap(2, 3)
        heat1.vals_[0,0] = 1.2
        heat1.counts_[0,0] = 3
        heat2.vals_[0,0] = 1.5
        heat2.counts_[0,0] = 2
        merged_heat = Heatmap.sum_heatmaps([heat1, heat2])
        self.assertEqual(merged_heat.vals_[0,0], 2.7)
        self.assertEqual(merged_heat.counts_[0,0], 5)
        self.assertEqual(merged_heat.vals_[0,1], 0.0)
        self.assertEqual(merged_heat.counts_[0,1], 0)

#=============================================================================

if __name__ == '__main__':
    unittest.main()

