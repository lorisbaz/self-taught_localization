import cPickle as pickle
import bsddb
import logging
import numpy as np
import os
import os.path
import random
import sys
import scipy.misc
import scipy.io
import scipy.sparse
import skimage.io
import xml.etree.ElementTree as ET
from vlg.util.parfun import *
from PIL import Image
from PIL import ImageDraw

from annotatedimage import *
from bbox import *
from heatmap import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from htmlreport import *
from util import *

class Params:
    def __init__(self):
        # mat_dir has to contain a file <key>.mat for each key, containing
        # the variables bboxes_GT, bboxes_SS, features_GT, features_SS
        self.mat_dir = ''

def get_filenames(params):
    """
    return the list of image-keys for train, validation, and test of VOC 2007
    """
    conf = params.conf
    keys = []
    for set_type in params.sets:
        fd = open(conf.pascal2007_sets_dir + '/' + set_type + '.txt')
        for line in fd:
            keys.append(line.strip())
        fd.close()
    return keys

def pipeline(image_keys, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    # loop over the images
    for image_key in image_keys:
        # load the image
        image_file = conf.pascal2007_images_dir + '/' + image_key + \
                     conf.pascal2007_image_file_extension
        logging.info('***** Elaborating ' + os.path.basename(image_file))
        img = skimage.io.imread(image_file)
        img = skimage.img_as_ubyte(img)
        # create the AnnotatedImage
        anno = AnnotatedImage()
        anno.set_image(img)
        anno.image_name = image_key
        anno.crop_description = 'original'
        # read the ground truth XML file
        xmlfile = conf.pascal2007_annotations_dir + '/' + image_key + '.xml'
        xmldoc = ET.parse(xmlfile)
        xmlanno = xmldoc.getroot()
        size_width = int(xmlanno.find('size').find('width').text)
        assert size_width == anno.image_width
        size_height = int(xmlanno.find('size').find('height').text)
        assert size_height == anno.image_height
		# read the MAT file
        fname = '{0}/{1}.mat'.format(params.mat_dir, image_key)
        logging.info('Load the MAT file {0}'.format(fname))
        M = scipy.io.loadmat(fname)
        # create the features
        featdata = None
        featidx = {}
        # fill out the GT objects
        logging.info('Fill out the GT boxes fields')
        featidx_idx = 0
        for idx_cat in range(len(conf.pascal2007_classes)):
            bbsraw = M['bboxes_GT'][idx_cat][0]
            if bbsraw.size == 0:
                continue
            assert bbsraw.shape[1] == 4
            assert bbsraw.shape[0] == M['features_GT'][idx_cat][0].shape[1]
            category = conf.pascal2007_classes[idx_cat]
            ao = AnnotatedObject(category)
            for idx_bb in range(bbsraw.shape[0]):
                bbraw = bbsraw[idx_bb, :]
                bb = BBox(bbraw[0]-1, bbraw[1]-1, bbraw[2], bbraw[3])
                assert bb.xmax <= size_width
                assert bb.ymax <= size_height
                bb.normalize_to_outer_box(BBox(0,0,size_width,size_height))
                ao.bboxes.append(bb)
                featidx[bb.get_coordinates_str()] = featidx_idx
                featidx_idx += 1
            feat = M['features_GT'][idx_cat][0].T
            if featdata == None:
                featdata = feat
            else:
                featdata = scipy.sparse.vstack([featdata, feat])
            anno.gt_objects[category] = ao
        # fill out the SS bboxes and features fields
        logging.info('Fill out the SS boxes fields')
        ao = AnnotatedObject('none')
        for idx_bb in range(M['bboxes_SS'].shape[0]):
            bbraw = M['bboxes_SS'][idx_bb, :]
            bb = BBox(bbraw[0]-1, bbraw[1]-1, bbraw[2], bbraw[3])
            assert bb.xmax <= size_width
            assert bb.ymax <= size_height
            bb.normalize_to_outer_box(BBox(0,0,size_width,size_height))
            ao.bboxes.append(bb)
            featidx[bb.get_coordinates_str()] = featidx_idx
            featidx_idx += 1
        feat = M['features_SS'].T
        if featdata == None:
            featdata = feat
        else:
            featdata = scipy.sparse.vstack([featdata, feat])
        assert featdata.shape[0] == featidx_idx
        #assert featdata.shape[0] == len(featidx)
        anno.pred_objects['SELECTIVESEARCH'] = {}
        anno.pred_objects['SELECTIVESEARCH']['none'] = ao
        logging.info(str(anno))
        # add the features to the AnnotatedImage
        anno.features = {}
        anno.features['FeatureExtractorMatlabGTSS'] = {}
        anno.features['FeatureExtractorMatlabGTSS'][params.mat_dir] = {}
        anno.features['FeatureExtractorMatlabGTSS'][params.mat_dir]['featdata']\
                         = featdata
        anno.features['FeatureExtractorMatlabGTSS'][params.mat_dir]['featidx']\
                         = featidx
        # write the output file
        pkl_fname = '{0}/{1}.pkl'.format(params.output_dir, \
                        remove_slash_and_extension_from_image_key(image_key))
        pkl_file = open(pkl_fname, 'wb')
        logging.info('Writing the output file {0}'.format(fname))
        pickle.dump(anno, pkl_file, protocol=2)
        pkl_file.close()
        logging.info('End record')
    return 0

def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # load the filenames of the the images
    images = get_filenames(params)
    # we sort the list, and split it into chunks
    images = sorted(images)
    image_chunks = split_list(images, params.num_chunks)
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
        jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
    	parfun = ParFunAnthill(pipeline, time_requested = 2, \
            job_name=jobname)
    else:
        parfun = ParFunDummy(pipeline)
    if len(params.task) == 0:
        idx_to_process = range(len(image_chunks))
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        outputfile = params.output_dir + '/%05d'%i
        outputdb = outputfile + '.db'
        outputhtml = outputfile + '.html'
        parfun.add_task(image_chunks[i], params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
