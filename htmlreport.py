import base64
import numpy as np
import os
import skimage
import skimage.io
import skimage.transform
import tempfile

import util

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
            '<body '.format(title)
        self.bodyopt_ = ' onload="' # run scripts here
        self.body_ = '"> '
        self.javascript_ = ''
        self.footer_ = \
            '</body>'\
            '</html>'

    def add_image_embedded(self, img_o, proportion = 1.0, max_size = -1, \
                           text = '', bboxes = [], isgt = False):
        """
        Embedds an image in the Html page.
        'img_o' is a ndarray.
        'proportion' scales the image (1.0 = orginal size).
        'max_size' specifies the maximum size of the image edges.
                   If max_size > 0.0, 'proportion' is ignored and the image
                   is resized having the maximum edge of size 'max_size'.
        'text' is some text (or Html code) put under the image.
        'bboxes' is a list of BBox objects 
        'isgt': TODO LORIS. what is this? :)
        """
        # resize the image, and convert it to jpeg
        img = self.resize_image_(img_o, proportion, max_size)
        img_str = util.convert_image_to_jpeg_string(img)
        # convert the bytes to base64 and add the image to the html page
        img_str_base64 = base64.b64encode(img_str)
        src = 'data:image/jpg;base64,' + img_str_base64
        if len(bboxes)>0:
            self.add_image_support_bboxes_(img, src, text)
            bboxes = self.resize_bboxes_(img_o, img, bboxes)
            self.add_body_options_(bboxes, text, isgt)
        else:
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
        self.add_javascript_()
        return self.header_ + self.bodyopt_ + self.body_ + \
               self.javascript_ + self.footer_

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

    def resize_bboxes_(self, img_o, img, bboxes):
        proportion_w = np.shape(img)[1]/float(np.shape(img_o)[1])
        proportion_h = np.shape(img)[0]/float(np.shape(img_o)[0])
        for i in range(np.shape(bboxes)[1]):
            bboxes[0,i] = bboxes[0,i]*proportion_w
            bboxes[1,i] = bboxes[1,i]*proportion_h
            bboxes[2,i] = bboxes[2,i]*proportion_w
            bboxes[3,i] = bboxes[3,i]*proportion_h              
        return bboxes

    def add_image_support_bboxes_(self, img, src, text):
        self.body_+= \
             '<div style="float:left;width:{0}px;margin-right:5px;">'\
             '<canvas id="{1}" width="{2}" height="{3}" title="{4}"></canvas>'\
             '</br>{5}'\
             '</div>'\
             .format(img.shape[1], text, img.shape[1], img.shape[0], src, text)

    def add_image_support_(self, img, src, text):
        self.body_+= \
              '<div style="float:left;width:{0}px;margin-right:5px;">'\
              '<img width="{1}" height="{2}" src="{3}"/>{4}'\
              '</div>'\
              .format(img.shape[1], img.shape[1], img.shape[0], src, text)

    def add_javascript_(self):
        self.javascript_ = \
         '<script> '\
         'function drawEverything(bboxes,cavasname,color_str){' \
         '   var c=document.getElementById(cavasname);' \
         '   var ctx=c.getContext("2d");' \
         '   img = new Image;' \
         '   img.src = c.title;' \
         '   ctx.drawImage(img,0,0,c.height,c.width);' \
         '   for(var i=0; i<bboxes.length; i++){' \
         '       if ((i+1)%4==0) {' \
         '           ctx.lineWidth="1";' \
         '           ctx.strokeStyle=color_str;' \
         '           ctx.rect(bboxes[i-3],bboxes[i-2],bboxes[i-1],bboxes[i]);'\
         '           ctx.stroke();' \
         '           if (color_str=="red"){' \
         '              ctx.fillStyle = "rgba(255, 0, 0, 0.3)";' \
         '           }else{' \
         '              ctx.fillStyle = "rgba(0, 255, 0, 0.3)";' \
         '           }' \
         '           ctx.fillRect (bboxes[i-3],bboxes[i-2],' \
                     'bboxes[i-1],bboxes[i]);' \
         '      }'\
         '   }'\
         '}'\
         '</script>'

    def add_body_options_(self, bboxes, text, isgt='False'):
        if isgt:
            color = 'green'
        else:
            color = 'red'
        # make flat bboxes (javascript will go over a vector)
        bboxes_flat = bboxes.T.flatten()
        bboxes_str = '['
        for i in range(len(bboxes_flat)):
            if i<len(bboxes_flat):
                bboxes_str += str(bboxes_flat[i]) + ', '
            else:
                bboxes_str += str(bboxes_flat[i])
        bboxes_str += ']' 
        # add values to bodyopt
        self.bodyopt_+= 'drawEverything({0},\'{1}\',\'{2}\');' \
                        .format(bboxes_str, text, color)


