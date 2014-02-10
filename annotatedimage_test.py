
import unittest

from annotatedimage import *
from bbox import *
from heatmap import *


class AnnotatedImageTest(unittest.TestCase):
    """
    TODO. Write a proper unittest. This code is just an example.
    """
    def setUp(self):
        self.img_anno = AnnotatedImage()

    def tearDown(self):
        self.img_anno = None

    def test(self):
        img = skimage.io.imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        skimage.io.imshow(img)
        self.img_anno.set_image(img)
        img2 = self.img_anno.get_image()
        skimage.io.imshow(img2)    
        skimage.io.show()

#=============================================================================

if __name__ == '__main__':
    unittest.main()
