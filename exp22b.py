import cPickle as pickle
import bsddb
import glob
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import skimage.io
from vlg.util.parfun import *
from scipy.io import *

from annotatedimage import *
from bbox import *
from network import *
from configuration import *
from util import *
from stats import *

class Params:
    def __init__(self):
        # execute the extraction of the statistics from the images.
        # If false, we assumed that we did run the extraction before, and we
        # perform just the aggregation.
        self.run_stat_pipeline = True
        # Re-ranking variables: use the full-img classification score and
        self.full_img_class = False
        # If True, we select the subset of classes that overlap between
        # ILSVRC2012-class and ILSVRC2013-det
        self.select_subset_overlap_ilsvrc2012_ilsvrc2013 = False
        # ... consider the classication score given the GT label
        self.rerank_with_GT_label = False

def pipeline(inputdb, outputdb, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Init net
    conf = params.conf
    if params.select_subset_overlap_ilsvrc2012_ilsvrc2013:
        # Retrieve wnids (used to rule out GTs)
        locids, wnids_my_subset = \
                    get_wnids(conf.ilsvrc2013_classid_wnid_words_overlap)
    else:
        wnids_my_subset = []
    if params.classifier=='CAFFE':
        netParams = NetworkCaffe1114Params(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    net = Network.create_network(netParams)
    # Open the database
    logging.info('outputdb: ' + outputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        # Get img
        img = anno.get_image()
        # set GT
        if params.rerank_with_GT_label:
            GT_label =  anno.get_gt_label()
        else:
            GT_label = None
        for classifier in anno.pred_objects.keys():
            pred_objects = anno.pred_objects[classifier]
            anno.pred_objects[classifier] = reRank_pred_objects(pred_objects,\
                                            img, net, params.full_img_class,\
                                            GT_label)
        # adding object  to the database
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
            memory_requested=1, job_name=jobname)
    else:
        parfun = ParFunDummy(pipeline)
    if params.task==None or len(params.task)==0:
        idx_to_process = range(n_chunks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputfile = params.output_dir + '/%05d'%i
        outputdb = outputfile + '.db'
        parfun.add_task(inputdb, outputdb, params)
    if params.run_stat_pipeline:
        out = parfun.run()
        for i, val in enumerate(out):
            if val != 0:
                logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
