import cPickle as pickle
import bsddb
import glob
import logging
import numpy as np
import argparse
import os
import os.path
import sys
import scipy.misc
import scipy.io
import skimage.io
import tempfile
from vlg.util.parfun import *

from htmlreport import *
from annotatedimage import *
from configuration import *

class ExportAnnoImageParams:
    def __init__(self):
        # input directory where the annotatedimages are store (with results)
        self.input_dir = ''
        # output file to store topN bboxes
        self.output_textfile = 'out.txt'
        # topN results are stored in the <input_dir>/htmls/ if True
        self.generate_htmls = False
        # Name of the classifier/method used to generate the results
        self.name_pred_objects = ''
        # Max number of buonding boxes that are stored
        self.max_num_bboxes = 1000
        # Usual parameters to run the exp on the cluster
        self.run_on_anthill = True
        self.task = []

def pipeline_map(inputdb, outputhtml, params):
    conf = params.conf
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    out = []
    if params.generate_htmls:
        htmlres = HtmlReport() 
    for image_key in db_keys:
        logging.info('Elaborating key ' + image_key)
        anno_img = pickle.loads(db_input[image_key])
        assert anno_img.image_name == image_key
        # Generate html output
        if params.generate_htmls:
            textout, pred_objects = anno_img.export_pred_bboxes_to_text( \
               params.name_pred_objects, params.max_num_bboxes, \
               output_filtered_pred_obj = True)
            anno_img.pred_objects[params.name_pred_objects] = pred_objects
            logging.info('Generating HTML ' + outputhtml)
            # visualize the annotation to a HTML row
            htmlres.add_annotated_image_embedded(anno_img, \
                    img_max_size=params.html_max_img_size, \
                    heatmaps_view = False)
            htmlres.add_newline()
        else:
            textout = anno_img.export_pred_bboxes_to_text( \
               params.name_pred_objects, params.max_num_bboxes)
        out.append(textout)
    # write the HTML
    if params.generate_htmls:
        logging.info('Saving HTML ' + outputhtml)
        htmlres.save(outputhtml)
    # Close DB
    db_input.close()
    return out

def pipeline_reduce(texts, params):
    fd = open(params.output_dir + '/' + params.output_textfile, 'w')
    for t in texts:
        fd.write(t)
    fd.close()

def pipeline(params):
    # Map
    logging.info('Map')
    n_chunks = len(glob.glob(params.input_dir + '/*.db'))
    parfun = None
    params.html_dir = params.input_dir + '/htmls'
    if params.generate_htmls and \
                    not os.path.exists(params.html_dir):
        os.makedirs(params.html_dir)
    if params.run_on_anthill:
    	parfun = ParFunAnthill(pipeline_map, time_requested=1)
    else:
        parfun = ParFunDummy(pipeline_map)
    if params.task==None or len(params.task)==0:
        idx_to_process = range(n_chunks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdb = params.input_dir + '/%05d'%i + '.db'
        if params.generate_htmls:
            outputhtml = params.html_dir + '/%05d'%i + '.html'
        else:
            outputhtml = ''
        parfun.add_task(inputdb, outputhtml, params)
    out = parfun.run()
    out = [x for l in out for x in l]
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    # Reduce
    logging.info('Reduce')
    pipeline_reduce(out, params)

