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
from imgsegmentation import *
from configuration import *
from htmlreport import *
from util import *
from network import *

class Params:
    def __init__(self):
        pass

def pipeline(inputdb, outputdb, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    if params.classifier=='CAFFE':
        net = NetworkCaffe(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only)
    elif params.classifier=='DECAF':
        net = NetworkDecaf(conf.ilsvrc2012_decaf_model_spec, \
                           conf.ilsvrc2012_decaf_model, \
                           conf.ilsvrc2012_classid_wnid_words, \
                           center_only = params.center_only)
    segmenter = ImgSegm_ObfuscationSearch(net, params.ss_version, \
                                        params.min_sz_segm, params.topC)
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    label = 'none'
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()
        img_width, img_height = np.shape(img)[0:2]
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        # resize img to fit the size of the network
        image_resz = skimage.transform.resize(img,\
                                    (net.get_input_dim(), net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz) 
        # extract segments
        segment_lists = segmenter.extract(image_resz)
        # Convert the segmentation lists to BBoxes
        pred_bboxes_unnorm = segments_to_bboxes(segment_lists)
        # Normalize the bboxes
        pred_bboxes = []
        for j in range(np.shape(pred_bboxes_unnorm)[0]):
            pred_bboxes_unnorm[j].normalize_to_outer_box(BBox(0, 0, \
                                                img_width, img_height))
            pred_bboxes.append(pred_bboxes_unnorm[j])
        # store results
        anno.pred_objects['OBFUSCSEARCH'] = {}
        anno.pred_objects['OBFUSCSEARCH'][label] = pred_bboxes
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
    # change the protobuf file (for batch mode)
    filetxt = open(params.conf.ilsvrc2012_caffe_model_spec)
    # save new file locally
    if params.classifier=='CAFFE':
        params.conf.ilsvrc2012_caffe_model_spec = \
                                            'imagenet_deploy_tmp.prototxt'
        filetxtout = open(params.conf.ilsvrc2012_caffe_model_spec, 'w')
        l = 0
        for line in filetxt.readlines():
            #print line
            if l == 1: # second line contains num_dim
                line_out = 'input_dim: ' + str(params.batch_sz) + '\n'
            else:
                line_out = line
            filetxtout.write(line_out)        
            l += 1        
        filetxtout.close()
        filetxt.close() 
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
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(inputdb, outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')

