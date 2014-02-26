import cPickle as pickle
import bsddb
import glob
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import skimage.io
from vlg.util.parfun import *

from annotatedimage import *
from heatmap import *
from network import *
from configuration import *
from heatextractor import *
from htmlreport import *
from util import *

class Params:
    def __init__(self):
        pass

def pipeline(inputdb, outputdb, outputhtml, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    if params.run_extraction:
        net = NetworkDecaf(conf.ilsvrc2012_decaf_model_spec, \
                           conf.ilsvrc2012_decaf_model, \
                           conf.ilsvrc2012_classid_wnid_words, \
                           center_only = True)
        heatext = HeatmapExtractorBox(net, params.gray_par, \
                    confidence_tech = params.heatextractor_confidence_tech, \
                    area_normalization = params.heatextractor_area_normalization)
        db_input = bsddb.btopen(inputdb, 'c')
        db_output = bsddb.btopen(outputdb, 'c')
        db_keys = db_input.keys()
        # loop over the images
        for image_key in db_keys:
            # get database entry
            anno = pickle.loads(db_input[image_key])
            # get stuff from database entry
            img = anno.get_image()        
            logging.info('***** Elaborating ' + os.path.basename(anno.image_name))  
            # predict label for full image
            rep_vec = net.evaluate(img)
            pred_label = np.argmax(rep_vec)
            accuracy = np.max(rep_vec)
            pred_label = net.get_labels()[pred_label]
            # heatmaps extraction (with gt_label)
            heatmaps = heatext.extract(img, anno.get_gt_label()) 
            # add the heatmap obj to the annotation object 
            pred_tmp_object = {pred_label: AnnotatedObject(pred_label, accuracy)}
            pred_objects = {'DECAF': pred_tmp_object}
            for i in range(np.shape(heatmaps)[0]):
                heatmap_obj = AnnotatedHeatmap()
                heatmap_obj.heatmap = heatmaps[i].get_values()
                heatmap_obj.description = heatmaps[i].get_description()
                heatmap_obj.type = anno.get_gt_label()
                pred_objects['DECAF'][pred_label].heatmaps.append(heatmap_obj)
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
    # write the HTML
    logging.info('Composing ' + outputhtml)
    htmlres = HtmlReport()
    db_output = bsddb.btopen(outputdb, 'r')
    for key, value in db_output.iteritems():
        anno = pickle.loads(value)
        # visualize the annotation to a HTML row
        htmlres.add_annotated_image_embedded(anno)        
    htmlres.save(outputhtml)
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
    	parfun = ParFunAnthill(pipeline, time_requested = 10, \
                               job_name = 'J_{0}'.format(params.exp_name))
    else:
        parfun = ParFunDummy(pipeline)
    if len(params.task) == 0:
        idx_to_process = range(n_chunks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdb = params.input_dir + '/%05d'%i + '.db'
        outputfile = params.output_dir + '/%05d'%i
        outputdb = outputfile + '.db'
        outputhtml = outputfile + '.html'
        parfun.add_task(inputdb, outputdb, outputhtml, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
