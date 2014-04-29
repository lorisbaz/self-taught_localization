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
from configuration import *
from featextractor import *

class Params:
    def __init__(self):
        # Name of the classifier/method used to generate the results
        self.name_pred_objects = ''
        # save features to file 
        self.save_features_cache = True        
        # Max number of buonding boxes that are stored
        self.max_num_bboxes = 1000
        # Usual parameters to run the exp on the cluster
        self.run_on_anthill = True
        self.task = []
        # FeatureExtractorParams
        self.feature_extractor_params = None

def pipeline(inputdb, output_dir, params):
    # Display info machine for debugging purposes
    logging.info(os.uname())
    # Instantiate some objects, and open the database
    conf = params.conf
    assert params.feature_extractor_params != None
    feature_extractor_params = params.feature_extractor_params
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # create a file for each image
        if ('JPEG' in image_key) or ('png' in image_key): 
            pkl_filename = image_key.replace('JPEG','pkl')
        else:
            pkl_filename = image_key + '.pkl'
        pkl_filename = output_dir + '/' + pkl_filename
        pkl_file = open(pkl_filename, 'wb')
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        logging.info('***** Extracting features for ' + \
                                        os.path.basename(anno.image_name))
        # Register FeatureExtractorParams
        anno.register_feature_extractor(feature_extractor_params, \
                                        params.save_features_cache)
        # Filter the ouput pred bboxes (textout is NOT used!)
        if params.max_num_bboxes>0:
            textout, pred_objects = anno.export_pred_bboxes_to_text( \
                params.name_pred_objects, params.max_num_bboxes, \
                output_filtered_pred_obj = True)
            anno.pred_objects[params.name_pred_objects] = pred_objects
        # extract features for pred_objects and GT
        anno_pred_GT = [anno.pred_objects[params.name_pred_objects], \
                        anno.gt_objects]
        for anno_now in anno_pred_GT: 
            for label in anno_now.keys():
                anno.extract_features(anno_now[label].bboxes)
        # adding the AnnotatedImage with the heatmaps to the database 
        logging.info(str(anno))
        logging.info('Save annotated image to {0}'.format(pkl_filename))
        # write pkl representation to file
        pickle.dump(anno, pkl_file, protocol=2)
        pkl_file.close()
    return 0


def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # change the protobuf file (for batch mode)
    filetxt = open(params.conf.ilsvrc2012_caffe_model_spec)
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
        parfun.add_task(inputdb, params.output_dir, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')

