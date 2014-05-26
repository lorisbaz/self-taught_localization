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
import shutil
from vlg.util.parfun import *

from annotatedimage import *
from bbox import *
from configuration import *
from featextractor import *

class Params:
    def __init__(self):
        # Name of the classifier/method used to generate the results
        self.name_pred_objects = ''
        # save features to file
        self.save_features_cache = True
        # Usual parameters to run the exp on the cluster
        self.run_on_anthill = True
        self.task = []
        # FeatureExtractorParams
        self.feature_extractor_params = None
        # Compute the FULLIMAGE features
        self.full_image_boxes = True
        # Max num of bboxes selected for each method
        # 'CAFFE' means graysegm, graybox and sliding window
        self.max_num_bboxes = {'OBFSEARCH_GT': sys.maxint, \
                               'OBFSEARCH_TOPC': sys.maxint, \
                               'CAFFE': sys.maxint, \
                               'SELECTIVESEARCH': sys.maxint,\
                               'FULLIMAGE': sys.maxint}


def pipeline(inputdb, output_dir, params):
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
        #if ('JPEG' in image_key) or ('png' in image_key):
        #    pkl_filename = image_key.replace('JPEG','pkl')
        #else:
        pkl_filename = image_key + '.pkl'
        pkl_filename = output_dir + '/' + pkl_filename
        pkl_file = open(pkl_filename, 'wb')
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        logging.info('***** Extracting features for {0}, methods {1} '.format(\
                os.path.basename(anno.image_name), anno.pred_objects.keys()))
        # Register FeatureExtractorParams
        anno.register_feature_extractor(feature_extractor_params, \
                                        params.save_features_cache)
        # Extract features GT
        for label in anno.gt_objects.keys():
            anno.extract_features(anno.gt_objects[label].bboxes)
        # Filter the ouput pred bboxes (textout is NOT used!)
        if params.max_num_bboxes>0:
            for this_method in anno.pred_objects.keys():
                if isinstance(params.max_num_bboxes, dict):
                    if params.max_num_bboxes.has_key(this_method):
                        max_num_bboxes = params.max_num_bboxes[this_method]
                    else:
                        logging.info('Warning: params.max_num_bboxes has not'\
                                'the field {0}. We keep all the bboxes.'\
                                .format(this_method))
                        max_num_bboxes = sys.maxint
                else: # if a single value is provided, we use for all
                    max_num_bboxes = params.max_num_bboxes
                textout, pred_objects = anno.export_pred_bboxes_to_text( \
                            this_method, max_num_bboxes, \
                            output_filtered_pred_obj = True)
                anno.pred_objects[this_method] = pred_objects
        # extract features pred_objects from different METHOD(s)
        for this_method in anno.pred_objects.keys():
            for label in anno.pred_objects[this_method].keys():
                anno.extract_features(\
                                anno.pred_objects[this_method][label].bboxes)
        # adding the AnnotatedImage with the heatmaps to the database
        logging.info(str(anno))
        logging.info('Save annotated image to {0}'.format(pkl_filename))
        # write pkl representation to file
        pickle.dump(anno, pkl_file, protocol=2)
        pkl_file.close()
    return 0

def pipeline_merge(inputdbs, outputdb, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    # retrieve all the AnnotatedImages and images from the database
    dbs_input = []
    for inputdb in inputdbs:
        logging.info('Opening ' + inputdb)
        dbs_input.append(bsddb.btopen(inputdb, 'r'))
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = dbs_input[0].keys() # keys must be the same for all!
    # loop over the images
    for image_key in db_keys:
        anno_out = []
        for db_input in dbs_input:
            # get database entry
            anno = pickle.loads(db_input[image_key])
            # merge different anno images
            if anno_out == []:
                # init
                logging.info('***** Merging annotated images for ' + \
                                        os.path.basename(anno.image_name))
                anno_out = anno # this step copies the GT as well
            else:
                # merge the pred objects that are not present
                for this_method in anno.pred_objects.keys():
                    if not anno_out.pred_objects.has_key(this_method):
                        anno_out.pred_objects[this_method] = \
                                                anno.pred_objects[this_method]
                    else:
                        logging.info('Key {0} already present'.\
                                                    format(this_method))
            # Add full-image bboxes, if asked
            if params.full_image_boxes:
                anno.pred_objects['FULLIMAGE'] = {}
                for label in anno.gt_objects.keys():
                    annoobj = AnnotatedObject(label)
                    annoobj.bboxes = [BBox(0.0, 0.0, 1.0, 1.0)]
                    anno.pred_objects['FULLIMAGE'][label] = annoobj
        stuff = ['GT']
        stuff.extend(anno_out.pred_objects.keys())
        logging.info('- Method {0}'.format(stuff))
        # adding the AnnotatedImage with the heatmaps to the database
        logging.info(str(anno_out))
        db_output[image_key] = pickle.dumps(anno_out, protocol=2)
    # close all the input dbs
    for db_input in dbs_input:
        db_input.close()
    # save modifications to disk
    db_output.sync()
    db_output.close()
    return 0

def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # run the pipeline that merges different input dbs
    tmp_output_dir = params.output_dir + '_TMP'
    remove_tmp_output_dir = False
    if isinstance(params.input_dir, list):
        # create tmp output dir that will be destroyed at the end of the script
        if os.path.exists(tmp_output_dir) == False:
            os.makedirs(tmp_output_dir)
        # list the databases chuncks (use the first, have to be sync!)
        n_chunks = len(glob.glob(params.input_dir[0] + '/*.db'))
        # run the pipeline
        parfun = None
        if params.run_on_anthill:
            jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
            parfun = ParFunAnthill(pipeline_merge, time_requested=23, \
                                    job_name=jobname + '_merge')
        else:
            parfun = ParFunDummy(pipeline_merge)
        if len(params.task) == 0:
            idx_to_process = range(n_chunks)
        else:
            idx_to_process = params.task
        for i in idx_to_process:
            inputdbs = []
            for input_dir in params.input_dir:
                inputdbs.append(input_dir + '/%05d'%i + '.db')
            tmp_outputdb = tmp_output_dir + '/%05d'%i + '.db'
            parfun.add_task(inputdbs, tmp_outputdb, params)
        out = parfun.run()
        for i, val in enumerate(out):
            if val != 0:
                logging.info('Task {0} didn''t exit properly'.format(i))
        params.input_dir = tmp_output_dir
        remove_tmp_output_dir = True

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
    # delete tmp output dir (if exists)
    if remove_tmp_output_dir:
        logging.info('Removing temp directory {0}'.format(tmp_output_dir))
        shutil.rmtree(tmp_output_dir + '/')
    logging.info('End of the script')

