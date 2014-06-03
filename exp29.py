from util import *

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
from bbox import *
from bboxextractor import *
from heatmap import *
from imgsegmentation import *
from configuration import *
from htmlreport import *

class Params:
    def __init__(self):
        pass

def pipeline(inputdb, outputdb, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    images = []
    anno_imgs = []
    for image_key in db_keys:
        anno_img = pickle.loads(db_input[image_key])
        assert anno_img.image_name == image_key
        assert len(anno_img.pred_objects) == 0  # not necessary
        images.append( anno_img.get_image() )
        anno_imgs.append( anno_img )
    db_input.close()
    # extract the BING bboxes,
    # and append them to the 'BING' pred_objects
    logging.info('Extracting BING bounding boxes')
    bing_bboxes = bing(images)
    logging.info('Done.');
    bing_label = 'none'
    for idx in range(len(anno_imgs)):
        anno_img = anno_imgs[idx]
        anno_obj = AnnotatedObject(label=bing_label)
        anno_obj.bboxes.extend( bing_bboxes[idx] )
        anno_img.pred_objects['BING'] = {}
        anno_img.pred_objects['BING'][bing_label] = anno_obj
    # save the output database
    logging.info('Opening ' + outputdb)
    db_output = bsddb.btopen(outputdb, 'c')
    for idx in range(len(anno_imgs)):
        anno_img = anno_imgs[idx]
        value = pickle.dumps(anno_img, protocol=2)
        db_output[anno_img.image_name] = value
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
    n_chunks = len(glob.glob(params.input_dir + '/*.db'))
    # run the pipeline
    # TODO / NOTE: we use only gridiron, because the BING binary is not
    #              compatible for some reason with the katana nodes.
    parfun = None
    if params.run_on_anthill:
        jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
    	parfun = ParFunAnthill(pipeline, time_requested=1, \
            job_name=jobname, hostname_requested='gridiron*')
    else:
        parfun = ParFunDummy(pipeline)
    if params.task==None or len(params.task)==0:
        idx_to_process = range(n_chunks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputfile = params.output_dir + '/%05d'%i
        outputdb = outputfile + '.db'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
