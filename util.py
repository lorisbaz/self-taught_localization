import logging
# Log format (Note: this has to be here, because other import suppress it)
logging.basicConfig(level=logging.INFO, \
              format='[%(asctime)s %(filename)s:%(lineno)d] %(message)s', \
              datefmt='%m/%d/%Y %H:%M:%S')
import numpy as np
import os
import scipy.misc
import scipy.io
import skimage
import skimage.io
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
    
def split_list(l, num_chunks):
    """
    Split the given list 'l' into 'num_chunks' lists, trying to balance
    the number of elements in every sublist.
    Returns the list of sub-lists.
    """
    out = [[] for i in range(num_chunks)]
    numel = [0]*num_chunks
    idx = 0
    for i in range(len(l)):
        numel[idx] += 1
        idx += 1
        if idx >= num_chunks:
            idx = 0
    idx = 0
    for i in range(num_chunks):
        for j in range(numel[i]):
            out[i].append(l[idx])
            idx += 1
    return out

def selective_search(images, ss_version):
    """
    Extract the Selective Search subwindows from the given input images.
    This function runs the Matlab function selective_search.m.

    INPUT:
     images: list of N ndarray elements, each element is an image
     ss_version: The SelectiveSearch version to use. 
                 It is a string that could be either 'quality' or 'fast'.
                 Please refer to selective_search.m.
    OUTPUT:
     bboxes_all: list of N lists. The i-th sub-list is a list of BBox objects,
                 which are the SS subwindows associated to the i-th image.
                 Note that the confidence value is available, and that the
                 coordinates are 0-1 normalized.
    """
    # dump the images of the AnnotatedImages to temporary files
    img_temp_files = []
    for i in range(len(images)):
        (fd, tmpfile) = tempfile.mkstemp(suffix = '.bmp')
        os.close(fd)
        img_temp_files.append( tmpfile)
        img = skimage.io.imsave(tmpfile, images[i])
    # create temporary files for the .mat files
    mat_temp_files = []
    for i in range(len(images)):
        (fd, tmpfile) = tempfile.mkstemp(suffix = '.mat')
        os.close(fd)
        mat_temp_files.append( tmpfile )
    # run the Selective Search Matlab wrapper
    img_temp_files_cell = \
         '{' + ','.join("'{}'".format(x) for x in img_temp_files) + '}'
    mat_temp_files_cell = \
         '{' + ','.join("'{}'".format(x) for x in mat_temp_files) + '}'
    matlab_cmd = 'selective_search({0}, {1}, \'{2}\')'\
                  .format(img_temp_files_cell, mat_temp_files_cell, ss_version)
    command = "matlab -nojvm -nodesktop -r \"try; " + matlab_cmd + \
              "; catch; exit; end; exit\""
    logging.info('Executing command ' + command)
    if os.system(command) != 0:
        logging.error('Matlab SS script did not exit successfully!')
        return []
    # load the .mat files, and create the BBox objects
    bboxes_all = []
    for i in range(len(images)):
        try:
            mat = scipy.io.loadmat(mat_temp_files[i])
        except:
            logging.error('Exception while loading ' + mat_temp_files[i])
            bboxes_all.append( [] )
            continue    
        img_width = mat.get('img_width')[0,0]
        assert img_width > 0
        img_height = mat.get('img_height')[0,0]
        assert img_height > 0
        bboxes = mat.get('bboxes')
        assert bboxes.shape[1] == 4
        priority = mat.get('priority')
        assert priority.shape[1] == 1
        assert priority.shape[0] == bboxes.shape[0]
        bbs = []
        for j in range(bboxes.shape[0]):
            bb = BBox(bboxes[j,0]-1, bboxes[j,1]-1, bboxes[j,2], bboxes[j,3], \
                      priority[j,0])
            bb.normalize_to_outer_box(BBox(0, 0, img_width, img_height))
            bbs.append(bb)
        bboxes_all.append(bbs)
    assert len(bboxes_all) == len(images)
    # delete all the temporary files
    for i in range(len(images)):
        os.remove(img_temp_files[i])
        os.remove(mat_temp_files[i])
    # return
    return bboxes_all
