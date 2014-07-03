import logging
# Log format (Note: this has to be here, because other import suppress it)
LOG_FORMATTER = logging.Formatter( \
                '[%(asctime)s %(filename)s:%(lineno)d] %(message)s', \
                datefmt='%m/%d/%Y %H:%M:%S')
consoleHandler = logging.StreamHandler()
consoleHandler.setFormatter(LOG_FORMATTER)
logging.getLogger().addHandler(consoleHandler)
logging.getLogger().setLevel(logging.INFO)

# Display info machine for debugging purposes
import os
logging.info(os.uname())

import bsddb
import cPickle as pickle
import numpy as np
import os.path
import random
import scipy.misc
import scipy.io
import skimage
import skimage.io
import skimage.transform
import subprocess
import tempfile

from bbox import *

class TempFile:
    def __init__(self, mapped_file='', copy=True):
        """
        mapped_file is the file that you want to map. if copy=True, we copy
        this mapped file to the temporary location
        """
        # open the temporary file
        mapFileExtension = ''
        if mapped_file:
            mapFileName, mapFileExtension = os.path.splitext(mapped_file)
        (fd, tmpfilename) = tempfile.mkstemp(suffix=mapFileExtension)
        os.close(fd)
        self.tmpfilename = tmpfilename
        # set some class fields
        self.user = os.getenv('USER')
        self.mappedfilename = mapped_file
        # copy the mapped file, if requested
        if mapped_file and copy:
            command = 'scp {0}@anthill:{1} {2}'.format(\
                       self.user, mapped_file, tmpfilename)
            logging.info('Executing command ' + command)
            subprocess.check_call(command, shell=True)

    def get_temp_filename(self):
        return self.tmpfilename

    def close(self, copy=True):
        """
        If copy=True, we copy the temporary file (which is supposed to be
        modified to the mapped_file)
        """
        try:
            # copy the mapped file, if requested
            if copy:
                command = 'scp {0} {1}@anthill:{2}'.format(\
                    self.tmpfilename, self.user, self.mappedfilename)
                logging.info('Executing command ' + command)
                subprocess.check_call(command, shell=True)
        except:
            # remote the temporary file
            os.remove(self.tmpfilename)
            raise
        else:
            os.remove(self.tmpfilename)


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
        confidence = mat.get('confidence')
        assert confidence.shape[1] == 1
        assert confidence.shape[0] == bboxes.shape[0]
        bbs = []
        for j in range(bboxes.shape[0]):
            bb = BBox(bboxes[j,0]-1, bboxes[j,1]-1, bboxes[j,2], bboxes[j,3], \
                      confidence[j,0])
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

def bing(images):
    """
    Extract the BING subwindows from the given input images.
    This function runs the software
    /home/anthill/aleb/clients/bing/Src/BING_linux

    INPUT:
     images: list of N ndarray elements, each element is an image

    OUTPUT:
     bboxes_all: list of N lists. The i-th sub-list is a list of BBox objects,
                 which are the SS subwindows associated to the i-th image.
                 Note that the confidence value is available, and that the
                 coordinates are 0-1 normalized.
    """
    # dump the images of the AnnotatedImages to temporary files
    bboxes_all = []
    for i in range(len(images)):
        image = images[i]
        (fd, tmpfile_in) = tempfile.mkstemp(suffix = '.jpg')
        os.close(fd)
        img = skimage.io.imsave(tmpfile_in, image)
        # create temporary files for the output file
        (fd, tmpfile_out) = tempfile.mkstemp(suffix = '.txt')
        os.close(fd)
        # run the BING code
        command = '/home/anthill/aleb/clients/bing/Src/BING_linux '\
          'extract {0} {1}'.format(tmpfile_in, tmpfile_out);
        logging.info('Executing command ' + command)
        if os.system(command) != 0:
            logging.error('BING program did not exit successfully!')
            bboxes_all.append( [] )
            continue
        # load the output file, and create the BBox objects
        try:
            fd = open(tmpfile_out, 'r')
        except:
            logging.error('Exception while loading ' + mat_temp_files[i])
            bboxes_all.append( [] )
            continue
        # width:353
        field, delimiter, text = fd.readline().partition(':')
        assert field == 'width'
        img_width = int( text )
        assert img_width > 0
        # height:500
        field, delimiter, text = fd.readline().partition(':')
        assert field == 'height'
        img_height = int( text )
        assert img_height > 0
        # numBBoxes:1937
        field, delimiter, text = fd.readline().partition(':')
        assert field == 'numBBoxes'
        numBBoxes = int( text )
        assert numBBoxes > 0
        # -0.316005, 1, 257, 353, 500
        bbs = []
        for line in fd.readlines():
            fields = line.split(',')
            assert len(fields) == 5
            bb = BBox(int(fields[1].strip()) - 1, int(fields[2].strip()) - 1, \
                      int(fields[3].strip()), int(fields[4].strip()), \
                      float(fields[0].strip()))
            bb.normalize_to_outer_box(BBox(0, 0, img_width, img_height))
            bbs.append(bb)
        bboxes_all.append(bbs)
        fd.close()
        assert len(bbs) == numBBoxes
        # delete all the temporary files
        os.remove(tmpfile_in)
        os.remove(tmpfile_out)
    # return
    assert len(bboxes_all) == len(images)
    return bboxes_all

def reRank_pred_objects(pred_objects, image, net, full_img_class = False, \
                        GT_label = None):
    """
    Re-rank the confidence score of the pred_objects accordingly to the
    classification result of the bbox.
    """
    img_sz = np.shape(image)
    height = img_sz[0]
    width = img_sz[1]
    if full_img_class:
        image_rz = np.copy(image)
        # Resize crop
        image_rz = skimage.transform.resize(image_rz, \
                            (net.get_input_dim(), net.get_input_dim()))
        image_rz = skimage.img_as_ubyte(image_rz)
        # Compute Net max score
        net_feature_full = net.evaluate(image_rz)
        # Max
        if GT_label==None:
            id_max = np.argmax(net_feature_full)
        else:
            id_gt = net.get_label_id(GT_label)
    for label in pred_objects.keys():
        # Do it for each BBox
        for b in range(len(pred_objects[label].bboxes)):
            # project bbox to the image sizes
            ymin = np.int16(pred_objects[label].bboxes[b].ymin * height)
            ymax = np.int16(pred_objects[label].bboxes[b].ymax * height)
            xmin = np.int16(pred_objects[label].bboxes[b].xmin * width)
            xmax = np.int16(pred_objects[label].bboxes[b].xmax * width)
            # Extract crop in the image
            image_crop = np.copy(image[ymin:ymax, xmin:xmax])
            # Resize crop
            image_crop = skimage.transform.resize(image_crop, \
                                (net.get_input_dim(), net.get_input_dim()))
            image_crop = skimage.img_as_ubyte(image_crop)
            # Compute Net max score
            net_feature_vector = net.evaluate(image_crop)
            if full_img_class:
                if GT_label==None:
                    confidence = max(net_feature_full[id_max] - \
                                net_feature_vector[id_max], 0.0)
                else:
                    confidence = max(net_feature_full[id_gt] - \
                                net_feature_vector[id_gt], 0.0)
            else:
                confidence = np.max(net_feature_vector)
            # Overwrite the old confidence value
            pred_objects[label].bboxes[b].confidence = confidence

    return pred_objects


def segments_to_bboxes(segments):
    bboxes = []
    for s in range(np.shape(segments)[0]):
        for w in range(np.shape(segments[s])[0]):
            bboxes.append(segments[s][w]['bbox'])
    return bboxes


def randperm_deterministic(n):
    """
    Return a list, containing a deterministically-computed pseudo-random
    permutation of numbers from 0 to n-1

    NOTE: The determinist is guaranteed at least for Python 2.7
    """
    perm = range(n)
    random.seed(0)
    random.shuffle(perm)
    return perm

def dump_obj_to_file_using_pickle(obj, fname, mode='binary'):
    """ mode can be either 'binary' or 'txt' """
    fd = open(fname, 'wb')
    if mode == 'binary':
        pickle.dump(obj, fd, protocol=2)
    elif mode == 'txt':
        pickle.dump(obj, fd, protocol=0)
    else:
        raise ValueError('mode {0} not recognized'.format(mode))
    fd.close()

def load_obj_from_file_using_pickle(fname):
    fd = open(fname, 'r')
    obj = pickle.load(fd)
    fd.close()
    return obj

def load_obj_from_db(inputdb, idx=None, key=None):
    """
    inputdb is the .db file. You can specify either:
    - idx which is the index of the key in the file, xor
    - the key
    """
    assert (idx != None) or (key != None)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    if idx != None:
        key = db_keys[idx]
    anno_img = pickle.loads(db_input[key])
    db_input.close()
    return anno_img

def remove_slash_and_extension_from_image_key(image_key, remove_string = ''):
    """
    Replaces the slash character from image_key (if any) with an underscore.
    If remove_string is provided, also the string is removed.
    """
    new_image_key = image_key.replace('/','_')
    if (remove_string in image_key):
        new_image_key = new_image_key.replace(remove_string, '')
    return new_image_key

def get_wnids(classid_wnid_words_file):
    fd = open(classid_wnid_words_file)
    wnids = {}
    locids = []
    for line in fd:
        temp = line.strip().split('\t')
        locids.append(int(temp[0].strip()))
        wnids[temp[1].strip()] = temp[2].strip()
    fd.close()
    assert len(locids) == len(wnids)
    return locids, wnids

def compare_feature_vec(feature_vec_i, feature_vec_j, \
                        similarity = 'hist_intersection', normalize = True):
    """
    Compare two feature vectors with the selected similarity.
    """
    # normalize the features
    if normalize:
        feature_vec_i = feature_vec_i/np.float(np.sum(feature_vec_i))
        feature_vec_j = feature_vec_j/np.float(np.sum(feature_vec_j))
    # compute the distance
    if similarity == 'hist_intersection':
        out_dist = np.sum(np.minimum(feature_vec_i, feature_vec_j))
    else:
        raise NotImplementedError()
    return out_dist
