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
import time
from vlg.util.parfun import *

from annotatedimage import *
from bbox import *
from imgsegmentation import *
from configuration import *
from util import *
from network import *

class Params:
    def __init__(self):
        pass

def pipeline(inputdb, outputfile, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    wnids_my_subset = []
    if params.classifier=='CAFFE':
        netParams = NetworkCaffeParams(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    elif params.classifier=='DECAF':
        netParams = NetworkDecafParams(conf.ilsvrc2012_decaf_model_spec, \
                           conf.ilsvrc2012_decaf_model, \
                           conf.ilsvrc2012_classid_wnid_words, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    net = Network.create_network(netParams)
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    # loop over the images
    txt_content = ''
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        time_before = time.time()
        # resize img to fit the size of the network
        image_resz = skimage.transform.resize(img,\
                                    (net.get_input_dim(), net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz)
        logging.info(str(anno))
        # Perfor CNN inference
        caffe_rep_full = net.evaluate(image_resz)
        time_after = time.time()
        txt_content = txt_content + str(time_after-time_before) + '\n'
    # write the database
    fd = open(outputfile, 'w')
    fd.write(txt_content)
    fd.close()
    logging.info('Writing file ' + outputfile)
    return 0


def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
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
        outputdb = params.output_dir + '/%05d'%i + '.txt'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
