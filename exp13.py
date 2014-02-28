import cPickle as pickle
import bsddb
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import skimage.io
import glob
from vlg.util.parfun import *

from annotatedimage import *
from heatmap import *
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
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
    segmenter = ImgSegmFromMatFiles_List(conf.ilsvrc2012_segm_results_dir, \
                                         conf.ilsvrc2012_root_images_dir, \
                                         params.segm_type_load, \
                                         params.min_sz_segm, params.subset_par)
    heatext = HeatmapExtractorSegm_List(net, segmenter, \
                confidence_tech = params.heatextractor_confidence_tech, \
                area_normalization = params.heatextractor_area_normalization,\
                image_transform = params.segm_type_load)

    print outputdb
    db_input = bsddb.btopen(inputdb, 'r')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()        
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))  
        # sync segmentation loader  
        segmenter.set_segm_name(anno.image_name)
        anno.segmentation_name = segmenter.get_segm_name()
        # predict label for full image
        rep_vec = net.evaluate(img)
        pred_label = np.argmax(rep_vec)
        accuracy = np.max(rep_vec)
        pred_label = net.get_labels()[pred_label]
        # heatmaps extraction (with gt_label)
        heatmaps = heatext.extract(img, anno.get_gt_label()) 
        # add the heatmap obj to the annotation object 
        pred_tmp_object = {pred_label: AnnotatedObject(pred_label, accuracy)}
        pred_objects = {params.classifier: pred_tmp_object}
        for i in range(np.shape(heatmaps)[0]):
            heatmap_obj = AnnotatedHeatmap()
            heatmap_obj.heatmap = heatmaps[i].get_values()
            heatmap_obj.description = heatmaps[i].get_description()
            heatmap_obj.type = anno.get_gt_label()
            pred_objects[params.classifier][pred_label].\
                                        heatmaps.append(heatmap_obj)
        # note: for the next exp store only the avg heatmap
        anno.pred_objects = pred_objects
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
    	parfun = ParFunAnthill(pipeline, time_requested = 10, \
                               job_name = params.job_name)    	 
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
