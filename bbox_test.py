import unittest

import bbox


class BBoxTest(unittest.TestCase):
    def setUp(self):
        self.bbox = bbox.BBox(0.2, 0.3, 0.5, 0.6, 0.1)

    def tearDown(self):
        self.bbox = None

    def test_area(self):
        self.assertAlmostEqual(self.bbox.area(), 0.09, places=5)

    def test_normalize_to_outer_box(self):
        self.bbox.normalize_to_outer_box((2, 3))
        self.assertAlmostEqual(self.bbox.xmin, 0.2/3.0, places=5)
        self.assertAlmostEqual(self.bbox.ymin, 0.3/2.0, places=5)
        self.assertAlmostEqual(self.bbox.xmax, 0.5/3.0, places=5)
        self.assertAlmostEqual(self.bbox.ymax, 0.6/2.0, places=5)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
