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
    for label, obj in anno.gt_objects.iteritems():
        for bb in obj.bboxes:
            if obj.label == anno.get_gt_label():
                draw.rectangle([bb.xmin*anno.image_width, \
                                bb.ymin*anno.image_height, \
                                bb.xmax*anno.image_width-1, \
                                bb.ymax*anno.image_height-1], \
                               outline='red')
    del draw
    img.show()

def pipeline(images, outputdb, outputhtml, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    htmlres = HtmlReport()
    db = bsddb.btopen(outputdb, 'c')
    # loop over the images
    for image in images:
        image_wnid, image_file = image
        anno = AnnotatedImage()
        logging.info('***** Elaborating ' + os.path.basename(image_file))
        original_img = skimage.io.imread(image_file)
        # rescale the image
        img = resize_image_max_size(original_img, params.fix_sz)
        img = skimage.img_as_ubyte(img)
        # crop the image (if requested)
        if params.image_transformation == 'centered_crop':
            bbox_center_crop = get_center_crop(original_img)
            img = crop_image_center(img)
        elif params.image_transformation == 'original':
            bbox_center_crop = BBox(0, 0, img.shape[1], img.shape[0])
        else:
            raise ValueError('params.image_transformation not valid')
        anno.set_image(img)
        anno.image_name = os.path.basename(image_file)
        anno.crop_description = params.image_transformation
        gt_label = image_wnid.strip()
        anno.gt_objects[gt_label] = AnnotatedObject(gt_label, 1.0)
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
        #anno_objects_dict = {} # dictionary label --> AnnoObject
        for obj in objects:
            label = obj.find('name').text.strip()
            if label not in anno.gt_objects:
                # Note: this situation should never happen, and this code is
                # here for future usages. We add the AnnotatedObject, but set
                # the confidence to zero, so to differentiate this object anno
                # from the full-image object annotation (which has conf=1.0).
                anno.gt_objects[label] = AnnotatedObject(label, 0.0)
            bboxes = obj.findall('bndbox')
            for bbox in bboxes:
                xmin = int(bbox.find('xmin').text)
                ymin = int(bbox.find('ymin').text)
                xmax = int(bbox.find('xmax').text)
                ymax = int(bbox.find('ymax').text)
                bb = BBox(xmin-1, ymin-1, xmax, ymax)
                bb.normalize_to_outer_box(bbox_center_crop)
                bb.intersect(BBox(0.0, 0.0, 1.0, 1.0))
                # it can happen that the gt bbox is outside the center bbox
                if bb.area() > 0.0:
                    anno.gt_objects[label].bboxes.append(bb)
        # make sure that anno.gt_objects has exactly one element
        assert len(anno.gt_objects) == 1
        logging.info(str(anno))
        # visualize the annotation (just for debugging)
        if params.visualize_annotated_images:
            visualize_annotated_image(anno)
        # visualize the annotation to a HTML row
        htmlres.add_annotated_image_embedded(anno)
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
    # write the HTML
    htmlres.save(outputhtml)
    return 0


def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # load the filenames of the the images
    images = get_filenames(params)
    # we organize the images by class, and split the list into chunks
    images = sorted(images, key=lambda x: x[0])
    image_chunks = split_list(images, params.num_chunks)
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
    	parfun = ParFunAnthill(pipeline, time_requested = 2, \
                            job_name = 'Job_{0}'.format(params.exp_name))
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
