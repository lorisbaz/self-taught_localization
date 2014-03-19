import unittest
import skimage.io

import util


class UtilTest(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_selective_search(self):
        # Implementation notes:
        # to generate the tests, run manually selective_search.m,
        # put a breakpoint before the save, and do
        # bboxes = [bboxes(:,1)-1 , bboxes(:,2)-1 , bboxes(:,3) , bboxes(:,4)]
        # bboxes = [bboxes(:,1)./img_width , bboxes(:,2)./img_height , bboxes(:,3)./img_width , bboxes(:,4)./img_height]
        image = skimage.io.imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        images = [image]
        bboxes = util.selective_search(images, ss_version='quality')        
        self.assertAlmostEqual(len(bboxes[0]), 5290)
        # test first bbox
        self.assertAlmostEqual(bboxes[0][0].xmin, 0.614000000000000)
        self.assertAlmostEqual(bboxes[0][0].ymin, 0.285333333333333)
        self.assertAlmostEqual(bboxes[0][0].xmax, 0.900000000000000)        
        self.assertAlmostEqual(bboxes[0][0].ymax, 0.496000000000000)
        self.assertAlmostEqual(bboxes[0][0].confidence, 0.003179143813317)        
        # test second bbox
        self.assertAlmostEqual(bboxes[0][1].xmin, 0.0)
        self.assertAlmostEqual(bboxes[0][1].ymin, 0.0)
        self.assertAlmostEqual(bboxes[0][1].xmax, 1.0)        
        self.assertAlmostEqual(bboxes[0][1].ymax, 1.0)
        self.assertAlmostEqual(bboxes[0][1].confidence, 0.003873535513487)        

    def test_selective_search2(self):
        image = skimage.io.imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        images = [image, image]
        bboxes = util.selective_search(images, ss_version='fast')        
        self.assertAlmostEqual(len(bboxes[0]), 834)
        self.assertAlmostEqual(len(bboxes[1]), 834)
        # test first image, first bbox
        self.assertAlmostEqual(bboxes[0][0].xmin, 0.09)
        self.assertAlmostEqual(bboxes[0][0].ymin, 0.0)
        self.assertAlmostEqual(bboxes[0][0].xmax, 1.0)        
        self.assertAlmostEqual(bboxes[0][0].ymax, 0.288)
        self.assertAlmostEqual(bboxes[0][0].confidence, 0.002731697745111)        
        # test second image, first bbox
        self.assertAlmostEqual(bboxes[1][0].xmin, 0.09)
        self.assertAlmostEqual(bboxes[1][0].ymin, 0.0)
        self.assertAlmostEqual(bboxes[1][0].xmax, 1.0)        
        self.assertAlmostEqual(bboxes[1][0].ymax, 0.288)
        self.assertAlmostEqual(bboxes[1][0].confidence, 0.002731697745111)        

#=============================================================================

if __name__ == '__main__':
    unittest.main()
