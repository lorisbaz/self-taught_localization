import base64
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
import tempfile

class HtmlReport():
    """
    Simple implementation to generate html reports
    """

    def __init__(self, title = 'HtmlReport'):
        self.header_ = \
            '<!DOCTYPE html>'\
            '<html>'\
            '<head>'\
            '<title>{0}</title>'\
            '</head>'\
            '<body>'.format(title)
        self.body_ = ''
        self.footer_ = \
            '</body>'\
            '</html>'

    def add_image_embedded(self, img, proportion = 1.0, max_size = -1, \
                           text = ''):
        """
        Embedds an image in the Html page.
        'img' is a ndarray.
        'proportion' scales the image (1.0 = orginal size).
        'max_size' specifies the maximum size of the image edges.
                   If max_size > 0.0, 'proportion' is ignored and the image
                   is resized having the maximum edge of size 'max_size'.
        'text' is some text (or Html code) put under the image.
        """
        # TODO this procedure is very hacky (how is that skimage does not
        #      accept a file handler?)
        # save a temporary filename, and read its bytes
        img = self.resize_image_(img, proportion, max_size)
        (fd, tmpfilename) = tempfile.mkstemp(suffix = '.jpg')
        os.close(fd)
        skimage.io.imsave(tmpfilename, img)
        fd = open(tmpfilename, 'rb')
        img_str = fd.read()
        fd.close()
        os.remove(tmpfilename)
        # convert the bytes to base64 and add the image to the html page
        img_str_base64 = base64.b64encode(img_str)
        src = 'data:image/jpg;base64,' + img_str_base64
        self.add_image_support_(img, src, text)

    def add_image(self, img_filename, proportion = 1.0, max_size = -1, \
                  text = ''):
        """
        Add an image, by creating a link pointing to 'img_filename'
        'proportion' scales the image (1.0 = orginal size).
        'text' is some text (or Html code) put under the image.
        """
        img = skimage.io.imread(img_filename)
        img = self.resize_image_(img, proportion, max_size)
        self.add_image_support_(img, img_filename, text)

    def add_text(self, text):
        """
        Add some text. 'text' can contain Html code.
        """
        self.body_ += text

    def add_newline(self):
        """
        Go to a new line.
        """
        self.body_ += '<br style="clear:both" />'

    def generate_html(self):
        """
        Returns the Html (a string).
        """
        return self.header_ + self.body_ + self.footer_

    def save(self, filename):
        """
        Write the Html page to a file.
        """
        fd = open(filename, 'w')
        fd.write( self.generate_html() )
        fd.close()

    def resize_image_(self, img, proportion, max_size):
        width = -1
        height = -1
        if max_size > 0:
            great_size = np.max(img.shape)
            proportion = max_size / float(great_size)
        width = int(img.shape[1] * float(proportion))
        height = int(img.shape[0] * float(proportion))    
        return skimage.transform.resize(img, (height, width))

    def add_image_support_(self, img, src, text):
        self.body_+= \
              '<div style="float:left;width:{0}px;margin-right:5px;">'\
              '<img width="{1}" height="{2}" src="{3}"/>{4}'\
              '</div>'\
              .format(img.shape[1], img.shape[1], img.shape[0], src, text)
