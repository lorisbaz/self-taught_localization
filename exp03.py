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
from heatmap import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from htmlreport import *
from util import *

class Params:
    def __init__(self):
        # If False, we randomly choose a subset of "num_classes" classes.
        # If True, we pick the first "num_classes" classes 
        #          according the official ILSVRC ordering.
        self.randomly_choose_classes = False
        # Select the subset that you want to load ('val' or 'train')
        self.subset = 'val'


def check_xmls(images, wnids, params, conf):
    """
    Make sure that the XML file specs are valid.
    Returns a vector of len(images) elements, with 1 if the XML is correct
    or 0 otherwise.
    """
    image_is_valid = [False]*len(images) 
    for i in range(len(images)):
        image = images[i][0]
        label_id = images[i][1]
        if params.subset == 'val':
            xmlfile = conf.ilsvrc2012_val_box_gt + '/' \
                    + os.path.basename(image).replace('.JPEG', '.xml')
        elif params.subset == 'train':
            xmlfile = conf.ilsvrc2012_train_box_gt + '/' + \
                    wnids[label_id-1] + '/' \
                    + os.path.basename(image).replace('.JPEG', '.xml')
        else:
            raise ValueError('Not existing subset. Select val or train.')
        # Check that the GT XML is present
        if not os.path.exists(xmlfile):
            logging.info('Warning: XML does not exist: {0}'.format(xmlfile))            
            continue
        # make sure the XML contains a single, valid label
        xmldoc = ET.parse(xmlfile)
        annotation = xmldoc.getroot()
        objects = annotation.findall('object')
        error_flag = False
        labels_for_this_image = {}
        for obj in objects:
            ll = obj.find('name').text.strip()
            labels_for_this_image[ll] = True
            # check if the label is valid
            if ll not in wnids:
                logging.info('Warning: Label not present in the label list'\
                                ' {0} '.format(xmlfile))
                error_flag = True
        # check that there is exactly one label
        if len(labels_for_this_image) != 1:
            logging.info('Warning: The number of labels is {0} for {1}'.format(\
                         len(labels_for_this_image), xmlfile))
            error_flag = True
        # declare this image valid or not
        if not error_flag:
            image_is_valid[i] = True
    # return
    return image_is_valid
            

def get_filenames(params):
    """
    Return a list of (wnid, filename) for all the files
    with label_id \in {1, ..., num_classes}
    """
    NUM_CLASSES = 1000
    conf = params.conf
    wnids = []
    if params.subset == 'val':
        images_set = conf.ilsvrc2012_val_images
        labels_set = conf.ilsvrc2012_val_labels
    elif params.subset == 'train':
        images_set = conf.ilsvrc2012_train_images
        labels_set = conf.ilsvrc2012_train_labels
    else:
        raise ValueError('Not existing subset. Select val or train.')    
    fd = open(conf.ilsvrc2012_classid_wnid_words)
    for line in fd:
        temp = line.strip().split('\t')
        wnids.append(temp[1].strip())
    fd.close()
    images = []
    labels_id = []
    fd = open(images_set)
    for line in fd:
        images.append(conf.ilsvrc2012_root_images_dir + '/' + line.strip())
    fd.close()
    fd = open(labels_set)
    for line in fd:
        labels_id.append(int(line.strip()))
    fd.close()
    assert len(images) == len(labels_id)
    # create a list of (image, label)
    image_labels = []
    for i in range(len(images)):
        image_labels.append( (images[i], labels_id[i]) )
    assert len(images) == len(image_labels)
    # HACK DUE TO BUGS IN THE IMAGENET XML SPECS
    if params.subset == 'train':
        image_chunks = split_list(image_labels, params.num_chunks)
        parfun = None
        if params.run_on_anthill:
            parfun = ParFunAnthill(check_xmls, time_requested=2, job_name='JobTEMP')
        else:
            parfun = ParFunDummy(check_xmls)
        idx_to_process = range(len(image_chunks))
        for i in idx_to_process:
            parfun.add_task(image_chunks[i], wnids, params, conf)
        out = parfun.run()
        # join the responces
        valid = list(itertools.chain(*out))
        assert len(valid)==len(images)
        images2 = [images[i] for i in range(len(images)) if valid[i]]
        labels_id2 = [labels_id[i] for i in range(len(labels_id)) if valid[i]]
        images = images2
        labels_id = labels_id2
        assert len(images) == len(labels_id)
    # define how many and which classes to keep, and how many imgs per class
    num_images_classes = [0]*NUM_CLASSES
    num_classes = params.num_classes
    if num_classes <= 0:
        num_classes = NUM_CLASSES
    num_images_per_class = params.num_images_per_class
    if num_images_per_class <= 0:
        num_images_per_class = sys.maxint
    classes_to_keep = range(1, NUM_CLASSES+1)
    if params.randomly_choose_classes:
        idxperm = randperm_deterministic(NUM_CLASSES)
        classes_to_keep = \
           [classes_to_keep[idxperm[i]] for i in range(NUM_CLASSES)]
    classes_to_keep = classes_to_keep[0:params.num_classes]
    # return only the correct labels, and at most num_images_per_class
    out = []
    for i in range(len(images)):
        num_images_classes[labels_id[i]-1] += 1
        if labels_id[i] not in classes_to_keep \
            or num_images_classes[labels_id[i]-1] > num_images_per_class:
            continue
        out.append( (wnids[labels_id[i]-1], images[i]) )
    # check if a class is with low num of images
    for i in range(len(num_images_classes)):
        if num_images_classes[i] < num_images_per_class and \
            i in classes_to_keep:
            logging.info('Warning. Class {0} with number of images per class lower' \
                            ' than {1}.'.format(wnids[i]), \
                            params.num_images_per_class)
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
        if params.subset == 'val':
            xmlfile = conf.ilsvrc2012_val_box_gt + '/' \
                    + os.path.basename(image_file).replace('.JPEG', '.xml') 
        elif params.subset == 'train':
            xmlfile = conf.ilsvrc2012_train_box_gt + '/' + gt_label + '/' \
                    + os.path.basename(image_file).replace('.JPEG', '.xml')
        else:
            raise ValueError('Not existing subset. Select val or train.')
        # Check if XML file exists
        if os.path.exists(xmlfile) == True:
            xmldoc = ET.parse(xmlfile)
            annotation = xmldoc.getroot()
            size_width = int(annotation.find('size').find('width').text)
            assert size_width == original_img.shape[1]
            size_height = int(annotation.find('size').find('height').text)
            assert size_height == original_img.shape[0]
            objects = annotation.findall('object')
            #anno_objects_dict = {} # dictionary label --> AnnoObject
            error_flag = False
            for obj in objects:
                label = obj.find('name').text.strip()
                if label not in anno.gt_objects:
                    # HACKS DUE TO SOME BUGS IN THE IMAGENET XML SPECS
                    #anno.gt_objects[label] = AnnotatedObject(label, 0.0)
                    logging.info('Warning: Label not present in the label list'\
                                    ' {0} '.format(xmlfile))
                    error_flag = True
                    assert 0 # THIS SHOULD NEVER HAPPEN
                    continue
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
            if error_flag:            
                logging.info('Warning: Image {0} not included'.format(xmlfile))
            else:
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
    logging.info('Get the filenames')
    images = get_filenames(params)
    # we organize the images by class, and split the list into chunks
    logging.info('Create the chunks')
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
