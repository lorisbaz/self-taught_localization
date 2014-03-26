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
    return the list of image-keys for train, validation, and test of VOC 2007
    """
    conf = params.conf
    keys = []
    for set_type in params.sets:
        fd = open(conf.pascal2007_sets_dir + '/' + set_type + '.txt')
        for line in fd:
            keys.append(line.strip())
        fd.close()
    return keys

def pipeline(image_keys, outputdb, outputhtml, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    htmlres = HtmlReport()
    db = bsddb.btopen(outputdb, 'c')
    # loop over the images
    for image_key in image_keys:
        # load the image
        image_file = conf.pascal2007_images_dir + '/' + image_key + \
                     conf.pascal2007_image_file_extension
        logging.info('***** Elaborating ' + os.path.basename(image_file))
        img = skimage.io.imread(image_file)
        img = skimage.img_as_ubyte(img)
        # create the AnnotatedImage
        anno = AnnotatedImage()
        anno.set_image(img)
        anno.image_name = image_key
        anno.crop_description = 'original'
        # read the ground truth XML file
        xmlfile = conf.pascal2007_annotations_dir + '/' + image_key + '.xml'
        xmldoc = ET.parse(xmlfile)
        xmlanno = xmldoc.getroot()
        size_width = int(xmlanno.find('size').find('width').text)
        assert size_width == anno.image_width
        size_height = int(xmlanno.find('size').find('height').text)
        assert size_height == anno.image_height
        xmlobjects = xmlanno.findall('object')
        #anno_objects_dict = {} # dictionary label --> AnnoObject
        for xmlobj in xmlobjects:
            difficult = int(xmlobj.find('difficult').text.strip())
            if difficult==1 and params.include_difficult_objects==False:
                continue
            label = xmlobj.find('name').text.strip()
            if label not in anno.gt_objects:
                anno.gt_objects[label] = AnnotatedObject(label)
            bbox = xmlobj.find('bndbox')
            xmin = int(bbox.find('xmin').text)
            ymin = int(bbox.find('ymin').text)
            xmax = int(bbox.find('xmax').text)
            ymax = int(bbox.find('ymax').text)
            bb = BBox(xmin-1, ymin-1, xmax, ymax)
            bb.normalize_to_outer_box(BBox(0,0,size_width,size_height))
            anno.gt_objects[label].bboxes.append(bb)
        logging.info(str(anno))
        # visualize the annotation to a HTML row
        htmlres.add_annotated_image_embedded(anno)
        htmlres.add_newline()
        # adding the AnnotatedImage to the database
        logging.info('Adding the record to the database')
        value = pickle.dumps(anno, protocol=2)
        db[image_key] = value
        logging.info('End record')
    # write the database
    logging.info('Writing file ' + outputdb)
    db.sync()
    db.close()
    # write the HTML
    htmlres.save(outputhtml)
    return 0

def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # load the filenames of the the images
    images = get_filenames(params)
    # we sort the list, and split it into chunks
    images = sorted(images)
    image_chunks = split_list(images, params.num_chunks)
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
        jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
    	parfun = ParFunAnthill(pipeline, time_requested = 2, \
            job_name=jobname)
    else:
        parfun = ParFunDummy(pipeline)
    if len(params.task) == 0:
        idx_to_process = range(len(image_chunks))
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        outputfile = params.output_dir + '/%05d'%i
        outputdb = outputfile + '.db'
        outputhtml = outputfile + '.html'
        parfun.add_task(image_chunks[i], outputdb, outputhtml, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
