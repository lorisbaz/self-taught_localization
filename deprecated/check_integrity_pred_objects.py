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

def pipeline(inputdb, params):
    # Instantiate some objects, and open the database
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
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
        # Assuming one label!!!!
        this_label = anno.pred_objects[classifier_name].keys()[0]
        if anno.pred_objects[classifier_name][this_label].bboxes == []:
            not_good_images.append(image_key)
        total_imgs += 1
        logging.info(str(anno))
    db_input.close()
    return (total_imgs, not_good_images)

def run_exp(params):
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
        parfun.add_task(inputdb, params)
    out = parfun.run()
    # reducer
    logging.info('Applying Reducer --- Aggregating Results')
    n_tot = 0
    list_corrupted = []
    for n_examples, list_examples in out:
        n_tot += n_examples
        list_corrupted.extend(list_examples)
    logging.info('Number of corrupted {0}/{1}, i.e., {2}%'.\
        format(len(list_corrupted), n_tot, \
                    len(list_corrupted)/float(n_tot)*100))
    logging.info('End of the script')

