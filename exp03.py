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

def pipeline(images, outputdb, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    print outputdb
    db = bsddb.btopen(outputdb, 'c')
    # loop over the images
    for image in images:
        image_wnid, image_file = image
        anno = AnnotatedImage()
        logging.info('***** Elaborating ' + os.path.basename(image_file))
        original_img = skimage.io.imread(image_file)
        # rescale the image (if necessary), and crop it to the central region
        img = resize_image_max_size(original_img, params.fix_sz)
        img = skimage.img_as_ubyte(img)
        bbox_center_crop = get_center_crop(img)
        img = crop_image_center(img)
        anno.set_image(img)
        anno.image_name = os.path.basename(image_file)
        anno.crop_description = 'central crop'
        anno.gt_label = image_wnid.strip()
        # read the ground truth XML file
        xmlfile = conf.ilsvrc2012_val_box_gt + '/' \
                    + os.path.basename(image_file).replace('.JPEG', '.xml')
        xmldoc = ET.parse(xmlfile)
        annotation = xmldoc.getroot()
        size_width = int(annotation.find('size').find('width').text)
        assert size_width == original_img.shape[1]
        size_height = int(annotation.find('size').find('height').text)
        assert size_height == original_img.shape[0]
        objects = annotation.findall('object')
        anno_objects_dict = {} # dictionary label --> AnnoObject
        for obj in objects:
            label = obj.find('name').text.strip()
            if label not in anno_objects_dict:
                anno_objects_dict[label] = AnnotatedObject()
                anno_objects_dict[label].label = label
            bboxes = obj.findall('bndbox')
            for bbox in bboxes:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                bb = BBox(xmin, ymin, xmax, ymax)
                bb.intersect(bbox_center_crop)
                bb.translate(bbox_center_crop.xmin, bbox_center_crop.ymin)
                anno_objects_dict[label].bboxes.append(bb)
        for key, value in anno_objects_dict.iteritems():
            anno.gt_objects.append(value)
        logging.info(str(anno))
        # visualize the annotation (just for debugging)
        if params.visualize_annotated_images:
            visualize_annotated_image(anno)
        # adding the AnnotatedImage to the database
        logging.info('Adding the record to the database')
        key = os.path.basename(image_file).strip()
        value = pickle.dumps(anno, protocol=2)
        db[key] = value
        logging.info('End record')
    # write the database
    logging.info('Writing file ' + outputdb)
    db.sync()
    db.close()
    return 0


def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # load the filenames of the the images and randomly shuffle it
    images = get_filenames(params)
    random.shuffle(images)
    image_chunks = split_list(images, params.num_chunks)
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
    	parfun = ParFunAnthill(pipeline, time_requested = 10)
    else:
        parfun = ParFunDummy(pipeline)
    for i in range(len(image_chunks)):
        outputdb = params.output_dir + '/%05d'%i + '.db'
        parfun.add_task(image_chunks[i], outputdb, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
