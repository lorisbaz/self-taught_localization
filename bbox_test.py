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
        # intersect 2 bbox (that do not have intesection)
        self.bflt.intersect(bbox.BBox(0.6, 0.8, 0.7, 0.9))
        self.assertAlmostEqual(self.bflt.xmin, 0.0, places=5)
        self.assertAlmostEqual(self.bflt.ymin, 0.0, places=5)
        self.assertAlmostEqual(self.bflt.xmax, 0.0, places=5)
        self.assertAlmostEqual(self.bflt.ymax, 0.0, places=5)

    def test_intersect2(self):
        bflt = bbox.BBox(0.96, 0.49, 1.2, 0.64)
        bflt.intersect(bbox.BBox(0.0, 0.0, 1.0, 1.0))
        self.assertAlmostEqual(bflt.xmin, 0.96, places=5)
        self.assertAlmostEqual(bflt.ymin, 0.49, places=5)
        self.assertAlmostEqual(bflt.xmax, 1.0, places=5)
        self.assertAlmostEqual(bflt.ymax, 0.64, places=5)

    def test_jaccard_similarity(self):
        s = self.bint.jaccard_similarity(bbox.BBox(70, 40, 120, 65))
        self.assertEqual(s, 300 / float(40*20 + 50*25 - 300))

        s = self.bflt.jaccard_similarity(bbox.BBox(0.6, 0.8, 0.7, 0.9))
        self.assertEqual(s, 0.0)

    def test_copy(self):
        bflt_copy = self.bflt.copy()
        bflt_copy.xmin = -1
        self.assertNotEqual(self.bflt.xmin, bflt_copy.xmin)

    def test_non_maxima_suppression(self):
        # empty case
        bb = bbox.BBox.non_maxima_suppression([], 0.5)
        self.assertEqual(bb, [])
        # None case
        bb = bbox.BBox.non_maxima_suppression(None, 0.5)
        self.assertEqual(bb, [])
        # 1-bbox case
        bboxes = [bbox.BBox(0.6, 0.8, 0.7, 0.9, confidence=0.5)]
        bb = bbox.BBox.non_maxima_suppression(bboxes, 0.5)
        self.assertEqual(len(bb), 1)
        self.assertEqual(bb[0].xmin, 0.6)
        self.assertEqual(bb[0].ymin, 0.8)
        self.assertEqual(bb[0].xmax, 0.7)
        self.assertEqual(bb[0].ymax, 0.9)
        self.assertEqual(bb[0].confidence, 0.5)
        # 3-bboxes example
        bboxes = [bbox.BBox(0.6, 0.8, 0.7, 0.9, confidence=0.5), \
                  bbox.BBox(0.6, 0.8, 0.7, 0.95, confidence=0.1), \
                  bbox.BBox(0.6, 0.8, 0.75, 0.9, confidence=0.7), \
                  bbox.BBox(0.1, 0.2, 0.4, 0.9, confidence=0.8) ]
        bb = bbox.BBox.non_maxima_suppression(bboxes, 0.3)
        self.assertEqual(len(bboxes), 4)
        self.assertEqual(bboxes[0].confidence, 0.5)
        self.assertEqual(bboxes[1].confidence, 0.1)
        self.assertEqual(bboxes[2].confidence, 0.7)
        self.assertEqual(bboxes[3].confidence, 0.8)        
        self.assertEqual(len(bb), 2)
        self.assertEqual(bb[0].confidence, 0.8)
        self.assertEqual(bb[0].xmax, 0.4)
        self.assertEqual(bb[1].confidence, 0.7)
        self.assertEqual(bb[1].xmax, 0.75)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
