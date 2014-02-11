import logging
# Log format (Note: this has to be here, because other import suppress it)
logging.basicConfig(level=logging.INFO, \
              format='%(asctime)s %(message)s %(funcName)s %(levelno)s', \
              datefmt='%m/%d/%Y %I:%M:%S %p')
import numpy as np
import os
import skimage
import skimage.transform
import tempfile

from bbox import *

def resize_image_max_size(img, fix_sz):
    """
    Return a resized version of the image, where the longest edge has 
    length 'fix_sz' pixels. The resizing mantains the proportion.
    """
    img = np.copy(img)
    great_size = np.max(img.shape)
    if great_size > fix_sz:
        proportion = fix_sz / float(great_size)
        width = int(img.shape[1] * float(proportion))
        height = int(img.shape[0] * float(proportion))    
        img = skimage.transform.resize(img, (height, width))
    return img

def crop_image_center(img):
    """
    Returns the crop of the image, made by taking the central region.
    """
    img = np.copy(img)
    bb = get_center_crop(img)
    img = img[bb.ymin:bb.ymax, bb.xmin:bb.xmax]
    return img

def get_center_crop(img):
    """
    Return a BBox representing the central crop of the image
    """
    if img.shape[0] < img.shape[1]:
        offset = (img.shape[1] - img.shape[0]) / 2
        return BBox(offset, 0, offset+img.shape[0], img.shape[0])
    else:
        offset = (img.shape[0] - img.shape[1]) / 2
        return BBox(0, offset, img.shape[1], offset+img.shape[1])

def convert_image_to_jpeg_string(img):
    # TODO this procedure is very hacky (how is that skimage does not
    #      accept a file handler?)
    # save a temporary filename, and read its bytes
    (fd, tmpfilename) = tempfile.mkstemp(suffix = '.jpg')
    os.close(fd)
    skimage.io.imsave(tmpfilename, img)
    fd = open(tmpfilename, 'rb')
    img_str = fd.read()
    fd.close()
    os.remove(tmpfilename)
    return img_str

def convert_jpeg_string_to_image(img_jpeg_string):
    # TODO this procedure is very hacky (how is that skimage does not
    #      accept a file handler?)
    # save a temporary filename, and read its bytes
    (fd, tmpfilename) = tempfile.mkstemp(suffix = '.jpg')
    os.close(fd)
    fd = open(tmpfilename, 'wb')
    fd.write(img_jpeg_string)
    fd.close()
    img = skimage.io.imread(tmpfilename)
    os.remove(tmpfilename)
    return img
    
