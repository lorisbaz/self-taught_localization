import os
import unittest
import skimage.io
import subprocess

import util
import vlg.util.parfun


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
        image = skimage.io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        images = [image]
        bboxes = util.selective_search(images, ss_version='quality')
        self.assertAlmostEqual(len(bboxes[0]), 5290)
        # test first bbox
        self.assertAlmostEqual(bboxes[0][0].xmin, 0.614000000000000)
        self.assertAlmostEqual(bboxes[0][0].ymin, 0.285333333333333)
        self.assertAlmostEqual(bboxes[0][0].xmax, 0.900000000000000)
        self.assertAlmostEqual(bboxes[0][0].ymax, 0.496000000000000)
        self.assertAlmostEqual(bboxes[0][0].confidence, -0.003179143813317)
        # test second bbox
        self.assertAlmostEqual(bboxes[0][1].xmin, 0.0)
        self.assertAlmostEqual(bboxes[0][1].ymin, 0.0)
        self.assertAlmostEqual(bboxes[0][1].xmax, 1.0)
        self.assertAlmostEqual(bboxes[0][1].ymax, 1.0)
        self.assertAlmostEqual(bboxes[0][1].confidence, -0.003873535513487)

    def test_selective_search2(self):
        image = skimage.io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        images = [image, image]
        bboxes = util.selective_search(images, ss_version='fast')
        self.assertAlmostEqual(len(bboxes[0]), 834)
        self.assertAlmostEqual(len(bboxes[1]), 834)
        # test first image, first bbox
        self.assertAlmostEqual(bboxes[0][0].xmin, 0.09)
        self.assertAlmostEqual(bboxes[0][0].ymin, 0.0)
        self.assertAlmostEqual(bboxes[0][0].xmax, 1.0)
        self.assertAlmostEqual(bboxes[0][0].ymax, 0.288)
        self.assertAlmostEqual(bboxes[0][0].confidence, -0.002731697745111)
        # test second image, first bbox
        self.assertAlmostEqual(bboxes[1][0].xmin, 0.09)
        self.assertAlmostEqual(bboxes[1][0].ymin, 0.0)
        self.assertAlmostEqual(bboxes[1][0].xmax, 1.0)
        self.assertAlmostEqual(bboxes[1][0].ymax, 0.288)
        self.assertAlmostEqual(bboxes[1][0].confidence, -0.002731697745111)

    def test_bing(self):
        image = skimage.io.imread('/home/anthill/aleb/clients/bing/VOC2007/JPEGImages/000001.jpg')
        images = [image]
        bboxes = util.bing(images)
        self.assertAlmostEqual(len(bboxes[0]), 1943)
        # test first image, first bbox
        self.assertAlmostEqual(bboxes[0][0].xmin, 0.0)
        self.assertAlmostEqual(bboxes[0][0].ymin, 256/500.0)
        self.assertAlmostEqual(bboxes[0][0].xmax, 353/353.0)
        self.assertAlmostEqual(bboxes[0][0].ymax, 500/500.0)
        self.assertAlmostEqual(bboxes[0][0].confidence, -0.317388)

    def test_randperm_deterministic(self):
        t = util.randperm_deterministic(5)
        self.assertAlmostEqual(t[0], 2)
        self.assertAlmostEqual(t[1], 0)
        self.assertAlmostEqual(t[2], 1)
        self.assertAlmostEqual(t[3], 3)
        self.assertAlmostEqual(t[4], 4)

#=============================================================================

def utiltesttempfile_support(mapped_file):
    # write something on the temporary file
    tmp = util.TempFile(mapped_file, copy=True)
    fd = open(tmp.get_temp_filename(), 'w')
    fd.write('THIS IS A TEST')
    fd.close()
    # close and mape the file
    tmp.close(copy=True)
    return 0

class UtilTestTempFile(unittest.TestCase):
    def setUp(self):
        self.mapped_file = '/home/ironfs/scratch/vlg/THIS_IS_A_TEST.dat'
        subprocess.check_call('touch {0}'.format(self.mapped_file), shell=True)

    def tearDown(self):
        os.remove(self.mapped_file)

    def test_tmpfile(self):
        """
        We try to map a file that is on ironfs.
        """
        tmp = util.TempFile(self.mapped_file, copy=True)
        # test get_temp_filename
        tmp.get_temp_filename()
        # write something on the temporary file
        fd = open(tmp.get_temp_filename(), 'w')
        fd.write('THIS IS A TEST')
        fd.close()
        # test close
        tmp.close(copy=True)
        # make sure we write the content back to the mapped file
        fd = open(self.mapped_file, 'r')
        line = fd.readline()
        fd.close()
        self.assertEqual(line, 'THIS IS A TEST')

    @unittest.skipIf(os.uname()[1] != 'anthill.cs.dartmouth.edu',
                     'Skipping TestParFunAnthill because we are not on Anthill')
    def test_tmpfile_cluster(self):
        pf = vlg.util.parfun.ParFunAnthill(utiltesttempfile_support, \
                            time_requested=1, memory_requested=1, \
                            hostname_requested = 'tanto*', ironfs=False)
        pf.add_task(self.mapped_file)
        out = pf.run()
        # make sure we write the content back to the mapped file
        fd = open(self.mapped_file, 'r')
        line = fd.readline()
        fd.close()
        self.assertEqual(line, 'THIS IS A TEST')

#=============================================================================

if __name__ == '__main__':
    unittest.main()
