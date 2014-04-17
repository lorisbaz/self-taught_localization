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

from annotatedimage import *
from configuration import *

class ExportAnnoImageParams:
    def __init__(self):
        self.input_dir = ''
        self.output_textfile = 'out.txt'
        self.name_pred_objects = ''
        self.max_num_bboxes = 1000
        self.run_on_anthill = True
        self.task = []

def pipeline_map(inputdb, params):
    conf = params.conf
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    out = []
    for image_key in db_keys:
        logging.info('Elaborating key ' + image_key)
        anno_img = pickle.loads(db_input[image_key])
        assert anno_img.image_name == image_key
        out.append( anno_img.export_pred_bboxes_to_text( \
               params.name_pred_objects, params.max_num_bboxes))
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
        parfun.add_task(inputdb, params)
    out = parfun.run()
    out = [x for l in out for x in l]
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    # Reduce
    logging.info('Reduce')
    pipeline_reduce(out, params)

