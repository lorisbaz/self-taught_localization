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

#import bsddb
#import cPickle as pickle
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


def read_mapping_file(mapping_file):
    pid = open(mapping_file)
    mapping = {}
    for line in pid.readlines():
        parsed_line = line.split("\t")
        if parsed_line[0] not in mapping.keys():
            mapping[parsed_line[0]] = {}
        mapping[parsed_line[0]][parsed_line[2]] = parsed_line[1] + " " + parsed_line[3]
    pid.close()
    return mapping
