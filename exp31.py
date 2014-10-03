import cPickle as pickle
import bsddb
import glob
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import scipy.io
import skimage.io
import tempfile
from vlg.util.parfun import *
import copy

from annotatedimage import *
from bbox import *
from imgsegmentation import *
from configuration import *
from util import *
from network import *
from reranker import *

class Params:
    def __init__(self):
        # Use the GT label instead of the set of topC labels predicted
        # by the classifier
        self.use_fullimg_GT_label = False
        # If True, we select the subset of classes that overlap between
        # ILSVRC2012-class and ILSVRC2013-det
        self.select_subset_overlap_ilsvrc2012_ilsvrc2013 = False

def pipeline(inputdb, outputdb, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    if params.select_subset_overlap_ilsvrc2012_ilsvrc2013:
        # Retrieve wnids (used to rule out GTs)
        locids, wnids_my_subset = \
                    get_wnids(conf.ilsvrc2013_classid_wnid_words_overlap)
    else:
        wnids_my_subset = []
    netParams = NetworkCaffeParams(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    net = Network.create_network(netParams)
    # instanciate reranker
    reranker = ReRankerNet(net)
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    if not params.use_fullimg_GT_label:
        classifier_name = 'OBFSEARCH_TOPC'
    else:
        classifier_name = 'OBFSEARCH_GT'
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        # resize img to fit the size of the network
        for this_label in anno.pred_objects[classifier_name].keys():
            if not params.use_fullimg_GT_label:
                assert this_label == 'none'
            pred_obj = copy.copy(anno.pred_objects[classifier_name][this_label])
            # NMS
            #pred_bboxes = BBox.non_maxima_suppression( \
            #                         pred_bboxes, params.nms_iou_threshold)
            # rerank
            for bb in pred_obj.bboxes:
                bb.confidence = reranker.evaluate(img, bb)
            anno.pred_objects[classifier_name][this_label] = pred_obj
            logging.info(str(anno))
            # adding the AnnotatedImage to the database
            logging.info('Adding the record to the database')
            value = pickle.dumps(anno, protocol=2)
            db_output[image_key] = value
            logging.info('End record')
    # write the database
    logging.info('Writing file ' + outputdb)
    db_output.sync()
    db_output.close()
    return 0

def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # list the databases chuncks
    n_chunks = len(glob.glob(params.input_dir + '/*.db'))
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
        jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
        parfun = ParFunAnthill(pipeline, time_requested=23, \
            job_name=jobname)
    else:
        parfun = ParFunDummy(pipeline)
    if len(params.task) == 0:
        idx_to_process = range(n_chunks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
