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

def visualize_heatmap_box(img, heatmaps, heatmap_avg, \
                          out_image_desc, out_bboxes):
    """
    Function useful for visualizing partial results during debuging. 
    """
    import matplotlib.pyplot as plt
    plt.subplot(3,4,1)
    plt.imshow(img)
    plt.title('Cropped img')
    height, width = np.shape(img)[0:2]
    for i in range(len(out_bboxes)):
        logging.info(str(out_bboxes[i]))
        out_bboxes[i].xmin = out_bboxes[i].xmin * width
        out_bboxes[i].xmax = out_bboxes[i].xmax * width
        out_bboxes[i].ymin = out_bboxes[i].ymin * height
        out_bboxes[i].ymax = out_bboxes[i].ymax * height         
        rect = plt.Rectangle((out_bboxes[i].xmin, out_bboxes[i].ymin), \
                              out_bboxes[i].xmax - out_bboxes[i].xmin, \
                              out_bboxes[i].ymax - out_bboxes[i].ymin, \
                              facecolor="#ff0000", alpha=0.4)
        plt.gca().add_patch(rect)
    for i in range(np.shape(heatmaps)[0]):
        plt.subplot(3,4,i+2)
        plt.imshow(heatmaps[i])
    plt.subplot(3,4,np.shape(heatmaps)[0]+2)
    plt.imshow(heatmap_avg)    
    plt.title('Avg Heatmap')
    plt.subplot(3,4,np.shape(heatmaps)[0]+3)
    plt.imshow(out_image_desc[0][0])
    plt.title(out_image_desc[0][1])      
    plt.subplot(3,4,np.shape(heatmaps)[0]+4)
    plt.imshow(out_image_desc[1][0])
    plt.title(out_image_desc[1][1])   
    
    
    plt.show()


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
        for classifier in anno.pred_objects.keys():
            for label in anno.pred_objects[classifier].keys():
                # Compute avg heatmap
                ann_heatmaps = anno.pred_objects[classifier][label].heatmaps
                heatmaps = [] 
                for j in range(len(ann_heatmaps)):
                    heatmaps.append(ann_heatmaps[j].heatmap)
                heatmap_avg = np.sum(heatmaps, axis=0)/np.shape(heatmaps)[0]
                # Extract Bounding box using heatmap
                out_bboxes, out_image_desc = \
                                bbox_extractor.extract(img, [heatmap_avg])
                # visualize partial results for debug
                #visualize_heatmap_box(img, heatmaps, heatmap_avg, \
                #                      out_image_desc, out_bboxes)
                # Save bboxes in the output database
                anno.pred_objects[classifier][label].bboxes = out_bboxes
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
