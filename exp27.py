import cPickle as pickle
import bsddb
import logging
import itertools
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
from configuration import *
from htmlreport import *
from util import *

class Params:
    def __init__(self):
        # Select the subset that you want to load ('val' or 'train')
        self.subset = 'val'

def load_filenames_mapper(curr_image_set, conf):
    images = []
    if not os.path.exists(curr_image_set):
        logging.info('Warning: the file {0} doesn\'t exist'.\
                                                format(curr_image_set))
    # Retrieve images list in the current file
    fd = open(curr_image_set)
    for line in fd:
        temp = line.strip().split()
        file_name = conf.ilsvrc2013_root_images_dir + '/' + temp[0].strip()
        if not (file_name in images):
            images.append(file_name)
        # check if exists
        if not os.path.exists(file_name):
            logging.info('Warning: the file {0} doesn\'t exist'.\
                                                format(file_name))
    fd.close()
    return images

def load_filenames_reducer(image_lists):
    image_list = []
    for list_i in image_lists:
        image_list.extend(list_i)
    # trick to get unique values from a list
    image_list = list(set(image_list))
    return image_list

def get_filenames(params):
    """
    Return a list of (wnid, filename) for all the files
    with label_id \in {1, ..., num_classes}
    """
    conf = params.conf
    if params.subset == 'val':
        images_set = conf.ilsvrc2013_val_images
    elif params.subset == 'train':
        images_set = conf.ilsvrc2013_train_images
    else:
        raise ValueError('Not existing subset. Select val or train.')
    if params.overlap_ilsvrc2012:
        classid_wnid_words_file = conf.ilsvrc2013_classid_wnid_words_overlap
    else:
        classid_wnid_words_file = conf.ilsvrc2013_classid_wnid_words
    locids, wnids = get_wnids(classid_wnid_words_file)
    #locids = range(1,201)
    NUM_CLASSES = len(locids)
    # Load the different txt files (name: <set>_<x>.txt, where
    #   <set>=params.subset and <x> is the class local id (between 1 to 200)
    parfun = None
    if params.run_on_anthill:
    	parfun = ParFunAnthill(load_filenames_mapper, time_requested = 2, \
                            job_name = 'Job_L_{0}'.format(params.exp_name))
    else:
        parfun = ParFunDummy(load_filenames_mapper)
    if len(params.task) == 0:
        idx_to_process = locids
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        parfun.add_task(images_set.format(i), conf)
    out = parfun.run()
    # reducer
    logging.info('Reducer that collects results from the mapper and aggregate.')
    images = load_filenames_reducer(out)
    logging.info('Total number of images: {0}'.format(len(images)))
    # Note: the check of the xml is not done anymore, because some images
    #   can contain either objects not in the list or some objects that
    #   we did not considered in the subset we selected. The pipeline will
    #   make sure to take only the gt_labels in wnids.
    return images

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

def pipeline(images, wnids, outputdb, outputhtml, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename), or the SAME CLASS (i.e. wnid is a constant).
    """
    # Instantiate some objects, and open the database
    conf = params.conf
    htmlres = HtmlReport()
    db = bsddb.btopen(outputdb, 'c')
    total_num_obj = 0
    total_num_obj_not = 0
    # loop over the images
    for image in images:
        image_file = image
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
            bbox_center_crop = BBox(0, 0, original_img.shape[1], \
                                    original_img.shape[0])
        else:
            raise ValueError('params.image_transformation not valid')
        anno.set_image(img)
        anno.image_name = os.path.basename(image_file)
        anno.crop_description = params.image_transformation
        # read the ground truth XML file
        if params.subset == 'val':
            xmlfile = conf.ilsvrc2013_val_box_gt + '/' \
                    + os.path.basename(image_file).replace('.JPEG', '.xml')
        elif params.subset == 'train':
            # 'extra' files are used as negatives during training
            if 'extra' not in image_file:
                prefolder = os.path.basename(image_file).split('_')
                xmlfile = conf.ilsvrc2013_train_box_gt + '/' + prefolder[0] + \
                    '/' + os.path.basename(image_file).replace('.JPEG', '.xml')
            else: # extra files have not xml!
                xmlfile = ''
        else:
            raise ValueError('Not existing subset. Select val or train.')
        # Check if XML file exists
        obj_included = 0
        obj_not_included = 0
        if os.path.exists(xmlfile) == True:
            xmldoc = ET.parse(xmlfile)
            annotation = xmldoc.getroot()
            size_width = int(annotation.find('size').find('width').text)
            assert size_width == original_img.shape[1]
            size_height = int(annotation.find('size').find('height').text)
            assert size_height == original_img.shape[0]
            objects = annotation.findall('object')
            #anno_objects_dict = {} # dictionary label --> AnnoObject
            for obj in objects:
                gt_label = obj.find('name').text.strip()
                bbox = obj.find('bndbox')
                # check that the label is in our list
                if gt_label in wnids.keys():
                    if gt_label not in anno.gt_objects.keys():
                        anno.gt_objects[gt_label] = \
                                        AnnotatedObject(gt_label, 1.0)
                    xmin = int(bbox.find('xmin').text)
                    ymin = int(bbox.find('ymin').text)
                    xmax = int(bbox.find('xmax').text)
                    ymax = int(bbox.find('ymax').text)
                    bb = BBox(xmin-1, ymin-1, xmax, ymax)
                    bb.normalize_to_outer_box(bbox_center_crop)
                    bb.intersect(BBox(0.0, 0.0, 1.0, 1.0))
                    # it can happen that the gt bbox is outside the center
                    if bb.area() > 0.0:
                        assert anno.gt_objects.has_key(gt_label)
                        anno.gt_objects[gt_label].bboxes.append(bb)
                        obj_included += 1
                else:
                    obj_not_included += 1
        if os.path.exists(xmlfile) == True or xmlfile == '':
            #not anymore: make sure that anno.gt_objects has exactly one element
            #assert len(anno.gt_objects) == 1
            logging.info(str(anno))
            if obj_included == 0:
                logging.info('Warning: Image {0} without bboxes, it might' \
                        ' be used for detection as negative'.format(xmlfile))
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
        else:
            logging.info('No XML file of GT bbox provided.')

        total_num_obj += obj_included
        total_num_obj_not += obj_not_included
    # write the database
    logging.info('Writing {0} images with {1} objects, excluding {2} objects'\
                 ' in the file {3}'.format(len(images), total_num_obj, \
                 total_num_obj_not,outputdb))
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
    logging.info('Get the filenames')
    filenames_file = params.output_dir + '/list_filenames.pkl'
    if os.path.exists(filenames_file):
        fd = open(filenames_file, 'r')
        images = pickle.load(fd)
        fd.close()
    else:
        images = get_filenames(params)
        fd = open(filenames_file, 'w')
        pickle.dump(images, fd)
        fd.close()
    # Retrieve wnids (used to rule out GTs)
    if params.overlap_ilsvrc2012:
        classid_wnid_words_file = \
                        params.conf.ilsvrc2013_classid_wnid_words_overlap
    else:
        classid_wnid_words_file = params.conf.ilsvrc2013_classid_wnid_words
    locids, wnids = get_wnids(classid_wnid_words_file)
    # we organize the images by class, and split the list into chunks
    logging.info('Create the chunks')
    # Splits in chuncks
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
        parfun.add_task(image_chunks[i], wnids, outputdb, outputhtml, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
