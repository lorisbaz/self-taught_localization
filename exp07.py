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
from htmlreport_pro import *

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
    htmlres = HtmlReportPro()
      
    print output_html
    db_input = bsddb.btopen(inputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()  
        gtlabel = anno.get_gt_label()
        desc = '{0}-{1}'.format(gtlabel, anno.image_name)
        logging.info('***** Elaborating HTML creation ' + \
                      os.path.basename(anno.image_name))  
        # Visualize results & Save results to html
        pred_bboxes = []
        for i in range(len(anno.pred_objects)):
            # Compute avg heatmap
            ann_heatmaps = anno.pred_objects[i].heatmaps
            label = anno.pred_objects[i].label
            confidence = anno.pred_objects[i].confidence
            if ann_heatmaps!=[]: # full img obj does not have heatmaps
                heatmaps = [] 
                for j in range(len(ann_heatmaps)):
                    heatmaps.append(ann_heatmaps[j].heatmap)
                    desc = 'heatmap'
                    htmlres.add_image_embedded( \
                                   heatmaps[j]*params.visual_factor, \
                                   max_size = params.html_max_img_size, \
                                   text = desc)

                heatmap_avg = np.sum(heatmaps, axis=0)/np.shape(heatmaps)[0]
                desc = 'AVG heatmap'
                htmlres.add_image_embedded(heatmap_avg*params.visual_factor, \
                             max_size = params.html_max_img_size, \
                             text = desc)
            # Retrieve predicted bboxes
            pred_bboxes.append(anno.pred_objects[i].bboxes)
             
        # Retrieve gt bboxes
        gt_bboxes = []
        for i in range(len(anno.gt_objects)):    
            gt_bboxes.append(anno.gt_objects[i].bboxes)

        # visualize partial results for debug
        if params.visualize_res and pred_bboxes!=[]:
            visualize_heatmap_box(img, heatmaps, heatmap_avg, \
                                  pred_bboxes, gt_bboxes)



        # create the HTML
        htmlres.add_image_embedded(img, max_size = params.html_max_img_size, \
                               text = desc, bboxes = pred_bboxes, \
                               isgt = False) # predicted bboxes
        htmlres.add_image_embedded(img, max_size = params.html_max_img_size, \
                               text = desc + ' GT', bboxes = gt_bboxes, \
                               isgt = True) # gt bboxes
                   
                    descr = 'Heatmap'
                    htmlres.add_image_embedded( \
                                   heatmaps[j]*params.visual_factor, \
                                   max_size = params.html_max_img_size, \
                                   text = desc)

                desc = 'AVG heatmap'
                htmlres.add_image_embedded(heatmap_avg*params.visual_factor, \
                             max_size = params.html_max_img_size, \
                             text = desc)
 

        

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
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
