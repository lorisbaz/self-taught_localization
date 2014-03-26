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
from heatmap import *
from configuration import *
from util import *
from stats import *

class Params:
    def __init__(self):
        pass

def merge_db(inputdbs, outputdb, params):
    """
    Merges many different databases into a single. Note that we generate new 
    classifier keys of the dictionary.
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    logging.info('outputdb: ' + outputdb)
    db_output = bsddb.btopen(outputdb, 'c')
    # Load db and keys (assume to have the same keys for all the DBs)
    db_input = bsddb.btopen(inputdbs[0], 'r')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno_out = pickle.loads(db_input[image_key])
        for classifier in anno_out.pred_objects.keys():
            anno_out.pred_objects[classifier] = {} # hack preserve compatib
        logging.info('***** Merging Pred bboxes ' + \
                     os.path.basename(anno_out.image_name))       
        # Merge db entries
        for indb in inputdbs:
            db_input = bsddb.btopen(indb, 'r')
            anno = pickle.loads(db_input[image_key])
            for classifier in anno.pred_objects.keys():
                anno_out.extend_pred_objects(anno, classifier)
        # adding stats to the database
            print str(anno.pred_objects['CAFFE'].keys())
        print str(anno_out.pred_objects['CAFFE'].keys())
        value = pickle.dumps(anno_out, protocol=2)
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
    n_chunks = len(glob.glob(params.input_dirs[0] + '/*.db'))
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
        jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
    	parfun = ParFunAnthill(merge_db, time_requested=10, \
            job_name=jobname)
    else:
        parfun = ParFunDummy(merge_db)
    if params.task==None or len(params.task)==0:
        idx_to_process = range(n_chunks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdbs = []
        for name in params.input_dirs:
            inputdbs.append(name + '/%05d'%i + '.db')
        outputfile = params.output_dir + '/%05d'%i
        outputdb = outputfile + '.db'
        parfun.add_task(inputdbs, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')

