import unittest

import bbox


class BBoxTest(unittest.TestCase):
    def setUp(self):
        self.bbox = bbox.BBox(0.2, 0.3, 0.5, 0.6, 0.1)

    def tearDown(self):
        self.bbox = None

    def test_area(self):
        self.assertAlmostEqual(self.bbox.area(), 0.09, places=5)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
