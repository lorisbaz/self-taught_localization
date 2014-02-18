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
from htmlreport import *

class Params:
    def __init__(self):
        pass

def visualize_heatmap_box(img, heatmaps, heatmap_avg, \
                          out_bboxes_all, gt_bboxes_all):
    """
    Function useful for visualizing partial results during debuging. 
    """
    import matplotlib.pyplot as plt
    plt.subplot(3,4,1)
    plt.imshow(img)
    plt.title('GT bboxes')
    height, width = np.shape(img)[0:2] 
    for j in range(len(gt_bboxes_all)):
        gt_bboxes = gt_bboxes_all[j]
        if gt_bboxes!=[]:
            for i in range(len(gt_bboxes)):
                logging.info(str(gt_bboxes[i]))
                rect = plt.Rectangle((gt_bboxes[i].xmin, gt_bboxes[i].ymin), \
                              gt_bboxes[i].xmax - gt_bboxes[i].xmin, \
                              gt_bboxes[i].ymax - gt_bboxes[i].ymin, \
                              facecolor="#00ff00", alpha=0.4)
                plt.gca().add_patch(rect)
    plt.subplot(3,4,2)
    plt.imshow(img)
    plt.title('Pred. bboxes')
    for j in range(len(out_bboxes_all)):
        out_bboxes = out_bboxes_all[j]
        if out_bboxes!=[]:
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
        plt.subplot(3,4,i+3)
        plt.imshow(heatmaps[i])
    plt.subplot(3,4,np.shape(heatmaps)[0]+3)
    plt.imshow(heatmap_avg)    
    plt.title('Avg Heatmap')
    plt.show()


def pipeline(inputdb, output_html, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    htmlres = HtmlReport()
      
    print output_html
    db_input = bsddb.btopen(inputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()  
        height, width = np.shape(img)[0:2]         
        gtlabel = anno.get_gt_label()
        logging.info('***** Elaborating HTML creation ' + \
                      os.path.basename(anno.image_name))  
        # Visualize GT, results & Save to html
        #gt_bboxes = []
        for label in anno.gt_objects.keys():
            ann_gt = anno.gt_objects[label]
            if ann_gt.bboxes!=[]:
                desc = 'GT-{0}-{1}'.format(ann_gt.label, anno.image_name)
                np_bbox = np.zeros((4,len(ann_gt.bboxes)))
                for j in range(len(ann_gt.bboxes)):
                    np_bbox[0,j] = ann_gt.bboxes[j].xmin * width
                    np_bbox[1,j] = ann_gt.bboxes[j].ymin * height
                    np_bbox[2,j] = (ann_gt.bboxes[j].xmax - \
                                    ann_gt.bboxes[j].xmin) * width
                    np_bbox[3,j] = (ann_gt.bboxes[j].ymax - \
                                    ann_gt.bboxes[j].ymin) * height
                htmlres.add_image_embedded(img, \
                            max_size = params.html_max_img_size, \
                            text = desc, bboxes = np_bbox, \
                            isgt = True) # gt bboxes
            #gt_bboxes.append(ann_gt.bboxes)

        #pred_bboxes = []
        for classifier in anno.pred_objects.keys():
            for label in anno.pred_objects[classifier].keys():
                # Load and visualize heatmaps
                ann_pred = anno.pred_objects[classifier][label]
                # Draw img and bboxes associated to the current avg heatmap 
                desc = '{0}-{1}'.format(ann_pred.label, anno.image_name)
                np_bbox = np.zeros((4,len(ann_pred.bboxes)))
                for j in range(len(ann_pred.bboxes)):
                    np_bbox[0,j] = ann_pred.bboxes[j].xmin * width
                    np_bbox[1,j] = ann_pred.bboxes[j].ymin * height
                    np_bbox[2,j] = (ann_pred.bboxes[j].xmax - \
                                    ann_pred.bboxes[j].xmin) * width
                    np_bbox[3,j] = (ann_pred.bboxes[j].ymax - \
                                    ann_pred.bboxes[j].ymin) * height 
                htmlres.add_image_embedded(img,\
                            max_size = params.html_max_img_size, \
                            text = desc, bboxes = np_bbox, \
                            isgt = False) # predicted bboxes     
                heatmaps = [] 
                for j in range(len(ann_pred.heatmaps)):
                    heatmaps.append(ann_pred.heatmaps[j].heatmap)
                    desc = 'Heatmap'
                    htmlres.add_image_embedded( \
                                   heatmaps[j]*params.visual_factor, \
                                   max_size = params.html_max_img_size, \
                                   text = desc)

                heatmap_avg = np.sum(heatmaps, axis=0)/np.shape(heatmaps)[0]
                desc = 'AVG heatmap'
                htmlres.add_image_embedded(heatmap_avg*params.visual_factor, \
                             max_size = params.html_max_img_size, \
                             text = desc)
                #pred_bboxes.append(ann_pred.bboxes)
                htmlres.add_newline()
             
        # visualize partial results for debug
        #    visualize_heatmap_box(img, heatmaps, heatmap_avg, \
        #                          pred_bboxes, gt_bboxes)
        htmlres.add_newline()
        logging.info(str(anno))

    # write the html
    logging.info('Writing file ' + output_html)
    # save html and exit
    htmlres.save(output_html)
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
            output_html = params.output_dir + '/%05d'%i + '.html'
            parfun.add_task(inputdb, output_html, params)
    else: # RUN just the selected task! (debug only)
        i = params.task
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputdb = params.output_dir + '/%05d'%i + '.html'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
