import unittest

import bbox


class BBoxTest(unittest.TestCase):
    def setUp(self):
        self.bflt = bbox.BBox(0.2, 0.3, 0.5, 0.6, 0.1)
        self.bint = bbox.BBox(60, 30, 100, 50)

    def tearDown(self):
        self.bflt = None
        self.bint = None

    def test_area(self):
        self.assertAlmostEqual(self.bflt.area(), 0.09, places=5)
        self.assertAlmostEqual(self.bint.area(), 40*20)

    def test_normalize_to_outer_box(self):
        # test bflt
        self.bflt.normalize_to_outer_box(bbox.BBox(0, 0, 3.0, 2.0))
        self.assertAlmostEqual(self.bflt.xmin, 0.2/3.0, places=5)
        self.assertAlmostEqual(self.bflt.ymin, 0.3/2.0, places=5)
        self.assertAlmostEqual(self.bflt.xmax, 0.5/3.0, places=5)
        self.assertAlmostEqual(self.bflt.ymax, 0.6/2.0, places=5)
        # test bint
        self.bint.normalize_to_outer_box(bbox.BBox(50, 10, 80, 60))
        self.assertAlmostEqual(self.bint.xmin, 10.0/30.0, places=5)
        self.assertAlmostEqual(self.bint.ymin, 20.0/50.0, places=5)
        self.assertAlmostEqual(self.bint.xmax, 50.0/30.0, places=5)
        self.assertAlmostEqual(self.bint.ymax, 40.0/50.0, places=5)

    def test_intersect(self):
        # test bflt
        self.bflt.intersect(bbox.BBox(0.15, 0.4, 0.35, 0.7))
        self.assertAlmostEqual(self.bflt.xmin, 0.2, places=5)
        self.assertAlmostEqual(self.bflt.ymin, 0.4, places=5)
        self.assertAlmostEqual(self.bflt.xmax, 0.35, places=5)
        self.assertAlmostEqual(self.bflt.ymax, 0.6, places=5)
        # test bint
        self.bint.intersect(bbox.BBox(50, 10, 80, 60))
        self.assertAlmostEqual(self.bint.xmin, 60, places=5)
        self.assertAlmostEqual(self.bint.ymin, 30, places=5)
        self.assertAlmostEqual(self.bint.xmax, 80, places=5)
        self.assertAlmostEqual(self.bint.ymax, 50, places=5)

    def test_jaccard_similarity(self):
        s = self.bint.jaccard_similarity(bbox.BBox(70, 40, 120, 65))
        self.assertEqual(s, 300 / float(40*20 + 50*25 - 300))

    def test_copy(self):
        bflt_copy = self.bflt.copy()
        bflt_copy.xmin = -1
        self.assertNotEqual(self.bflt.xmin, bflt_copy.xmin)
        
        

#=============================================================================

if __name__ == '__main__':
    unittest.main()
