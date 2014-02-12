import cPickle as pickle
import bsddb
import logging
import numpy as np
import os
import os.path
import random
import sys
import scipy.misc
import skimage.io
import xml.etree.ElementTree as ET
from vlg.util.parfun import *
from PIL import Image
from PIL import ImageDraw

from annotatedimage import *
from bbox import *
from heatmap import *
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from htmlreport import *
from util import *

class Params:
    def __init__(self):
        pass

def get_filenames(params):
    """
    Return a list of (wnid, filename) for all the files
    with label_id \in {1, ..., num_classe}
    """
    conf = params.conf
    wnids = []
    fd = open(conf.ilsvrc2012_classid_wnid_words)
    for line in fd:
        temp = line.strip().split('\t')
        wnids.append(temp[1].strip())
    fd.close()
    images = []
    labels_id = []
    fd = open(conf.ilsvrc2012_val_images)
    for line in fd:
        images.append(conf.ilsvrc2012_root_images_dir + '/' + line.strip())
    fd.close()
    fd = open(conf.ilsvrc2012_val_labels)
    for line in fd:
        labels_id.append(int(line.strip()))
    fd.close()
    assert len(images) == len(labels_id)
    # return only the labels 1..num_classes, and at most num_images_per_class
    num_images_classes = [0]*1000
    num_classes = params.num_classes
    if num_classes <= 0:
        num_classes = sys.maxint
    num_images_per_class = params.num_images_per_class
    if num_images_per_class <= 0:
        num_images_per_class = sys.maxint
    out = []
    for i in range(len(images)):
        num_images_classes[labels_id[i]-1] += 1
        if labels_id[i] > num_classes \
              or num_images_classes[labels_id[i]-1] > num_images_per_class:
            continue
        out.append( (wnids[labels_id[i]-1], images[i]) )
    return out

def visualize_annotated_image(anno):
    img = convert_jpeg_string_to_image(anno.image_jpeg)
    img = scipy.misc.toimage(img)
    draw = ImageDraw.Draw(img)
    for obj in anno.gt_objects:
        for bb in obj.bboxes:
            if obj.label == anno.gt_label:
                draw.rectangle([bb.xmin, bb.ymin, bb.xmax-1, bb.ymax-1], \
                               outline='red')
    del draw
    img.show()

def pipeline(outputdb, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    net = NetworkDecaf(conf.ilsvrc2012_decaf_model_spec, \
                       conf.ilsvrc2012_decaf_model, \
                       conf.ilsvrc2012_classid_wnid_words, \
                       center_only = True)
    segmenter = ImgSegmFromMatFiles(conf.ilsvrc2012_segm_results_dir, \
                                    conf.ilsvrc2012_root_images_dir, \
                                    params.fix_sz, \
                                    subset_par=True)
    heatext = HeatmapExtractorSegm(net, segmenter, \
                confidence_tech = params.heatextractor_confidence_tech, \
                area_normalization = params.heatextractor_area_normalization)

    print outputdb
    db = bsddb.btopen(outputdb, 'c')
    db_keys = db.keys()
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db[image_key])
        # get stuff from database entry
        img = anno.get_image()        
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))  
        # sync segmentation loader  
        segmenter.set_segm_name(anno.image_name)
        anno.segmentation_name = segmenter.segmname_
        # predict label for full image
        rep_vec = net.evaluate(img)
        pred_label = np.argmax(rep_vec)
        anno.pred_label = net.get_labels()[pred_label]
        # heatmaps extraction (with gt_label)
        heatmaps = heatext.extract(img, anno.gt_label) 
        # add the heatmap obj to the annotation object
        anno.pred_objects = AnnotatedObject()
        anno.pred_objects.label = anno.pred_label
        anno.pred_objects.heatmaps = heatmaps            
        logging.info(str(anno))
        # visualize the annotation (just for debugging)
        if params.visualize_annotated_images:
            visualize_annotated_image(anno)
        # adding the AnnotatedImage with the heatmaps to the database 
        logging.info('Adding the record to he database')
        key = os.path.basename(image_file).strip()
        value = pickle.dumps(anno, protocol=2)
        db[image_key] = value
        logging.info('End record')
    # write the database
    logging.info('Writing file ' + outputdb)
    #db.sync()
    #db.close()
    return 0


def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # list the databases chuncks
    n_chunks = len(os.listdir(params.output_dir + '/'))
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
    	parfun = ParFunAnthill(pipeline, time_requested = 10)
    else:
        parfun = ParFunDummy(pipeline)
    for i in range(n_chunks):
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
