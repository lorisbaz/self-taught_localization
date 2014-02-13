import cPickle as pickle
import bsddb
import logging
import numpy as np
import os
import os.path
#import random
import sys
import scipy.misc
import skimage.io
#import xml.etree.ElementTree as ET
from vlg.util.parfun import *
#from PIL import Image
#from PIL import ImageDraw

from annotatedimage import *
from bbox import *
from bboxextractor import *
from heatmap import *
from configuration import *
from util import *

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
    bbox_extractor = GrabCutBBoxExtractor( \
                            min_bbox_size = params.min_bbox_size, \
                            grab_cut_rounds = params.grab_cut_rounds, \
                            consider_pr_fg = params.consider_pr_fg)
   
    print outputdb
    db_input = bsddb.btopen(inputdb, 'c')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()        
        logging.info('***** Elaborating bounding boxes ' + \
                      os.path.basename(anno.image_name))  
        # Bbox extraction
        for i in range(len(anno.pred_objects)):
            # Compute avg heatmap
            ann_heatmaps = anno.pred_objects[i].heatmaps
            if ann_heatmaps!=[]: # full img obj does not have heatmaps
                heatmaps = [] 
                for j in range(len(ann_heatmaps)):
                    heatmaps.append(ann_heatmaps[j].heatmap)
                heatmap_avg = np.sum(heatmaps, axis=0)/np.shape(heatmaps)[0]
                # Extract Bounding box using heatmap
                out_bboxes, out_image_desc = \
                                bbox_extractor.extract(img, [heatmap_avg])
                # Save bboxes in the output database
        

        logging.info(str(anno))
        # adding the AnnotatedImage with the heatmaps to the database 
        logging.info('Adding the record to he database')
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
    n_chunks = len(os.listdir(params.input_dir + '/'))
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
    	parfun = ParFunAnthill(pipeline, time_requested = 10, \
                               job_name = params.job_name)
    else:
        parfun = ParFunDummy(pipeline)
    for i in range(n_chunks):
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
