import cPickle as pickle
import bsddb
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import skimage.io
from vlg.util.parfun import *

from annotatedimage import *
from bbox import *
from heatmap import *
from configuration import *
from util import *
from stats import *

class Params:
    def __init__(self):
        pass

def pipeline(inputdb, outputdb, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
      
    print outputdb
    db_input = bsddb.btopen(inputdb, 'c')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        anno.set_stats()
        # get stuff from database entry
        logging.info('***** Elaborating statistics ' + \
                      os.path.basename(anno.image_name))        
        # Flat Gt and Pred objects to BBoxes
        gt_bboxes, gt_lab = Stats.flat_anno_bboxes(anno.gt_objects)
        for classifier in anno.pred_objects.keys():
            pred_bboxes, pred_lab = Stats.flat_anno_bboxes( \
                                    anno.pred_objects[classifier])
            # Extract stats 
            stat_obj = Stats()
            stat_obj.compute_stats(pred_bboxes, gt_bboxes, \
                                   params.IoU_threshold)
            anno.stats[classifier] = stat_obj
        # adding stats to the database
        value = pickle.dumps(anno, protocol=2)
        db_output[image_key] = value                 
        logging.info(str(anno.stats[classifier]))
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
    n_chunks = len(os.listdir(params.input_dir + '/'))
    # run the pipeline
    parfun = None
    if (params.run_on_anthill and not(params.task>=0)):
    	parfun = ParFunAnthill(pipeline, time_requested = 10, \
                               job_name = params.job_name)
    else:
        parfun = ParFunDummy(pipeline)
    if not(params.task>=0):        
        for i in range(n_chunks):
            inputdb = params.input_dir + '/%05d'%i + '.db'
            outputdb = params.output_dir + '/%05d'%i + '.db'
            parfun.add_task(inputdb, outputdb, params)
    else: # RUN just the selected task! (debug only)
        i = params.task
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
