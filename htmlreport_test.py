from htmlreport import *
import unittest


class HtmlReportTest(unittest.TestCase):
    """
    TODO. Write a proper unittest. This code is just an example.
    """
    def setUp(self):
        self.report = HtmlReport()

    def tearDown(self):
        self.report = None

    def test(self):
        self.report.add_text('hello<br>')
        self.report.add_image('ILSVRC2012_val_00000001_n01751748.JPEG', \
                              proportion = 0.5, text = 'img proportion 0.5')
        img = skimage.io.imread('ILSVRC2012_val_00000001_n01751748.JPEG')
        self.report.add_image_embedded(img, proportion = 0.05, \
                                       text = 'img2 proportion 0.05')
        self.report.save('test.html')

#=============================================================================

if __name__ == '__main__':
    unittest.main()
