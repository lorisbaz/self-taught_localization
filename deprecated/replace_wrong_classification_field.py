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

from annotatedimage import *
from bbox import *
from configuration import *
from util import *

class Params:
    def __init__(self):
        self.exp_name_input = ''

def pipeline(inputdb, outputdb, params):
    # Instantiate some objects, and open the database
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    total_imgs = 0
    not_good_images = []
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        classifier_name = anno.pred_objects.keys()[0]
        # Adding fields and deeocopy
        assert classifier_name == params.field_to_be_replaced
        anno.pred_objects[params.replance_with_this] = \
                copy.deepcopy(anno.pred_objects[params.field_to_be_replaced])
        # Removing old field
        assert anno.pred_objects.has_key(params.replance_with_this)
        del anno.pred_objects[params.field_to_be_replaced]
        logging.info(str(anno))
        # Dump anno to the output
        db_output[image_key] = pickle.dumps(anno, protocol=2)
        logging.info('End of record')
    db_input.close()
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
        jobname = 'Job{0}'.format(params.exp_name_input).replace('exp','')
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

