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
from self_taught_loc import *

class Params:
    def __init__(self):
        # If ON, instead of obuscating the segment, we obfuscate the
        #  bbox sorrounding the segment.
        self.obfuscate_bbox = False
        # Use the GT label instead of the set of topC labels predicted
        # by the classifier
        self.use_fullimg_GT_label = False
        # If True, we select the subset of classes that overlap between
        # ILSVRC2012-class and ILSVRC2013-det
        self.select_subset_overlap_ilsvrc2012_ilsvrc2013 = False
        # For obf search. Use similarity instead of diversity
        self.function_stl = 'diversity'
        # If cnnfeature are used, we can include some padding to the
        # bbox where feature are extracted. value in [0.0, 1.0]
        self.padding = 0.0
        # Select a single parametrization of Feltz alg (i.e., single color space
        # and k)
        self.single_color_space = False
        # Remove obfuscation?
        self.no_obfuscation = False

def pipeline(inputdb, outputdb, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    if params.select_subset_overlap_ilsvrc2012_ilsvrc2013:
        # Retrieve wnids (used to rule out GTs)
        locids, wnids_my_subset = \
                    get_wnids(conf.ilsvrc2013_classid_wnid_words_overlap)
    else:
        wnids_my_subset = []
    # load mapping categories pascal
    mappingPASCAL = read_mapping_file(conf.pascal2007_mapping_file)
    if params.classifier=='CAFFE':
        netParams = NetworkCaffe1114Params(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    net = Network.create_network(netParams)
    # choose segmentation method (Matlab wrapper Felz through SS)
    img_segmenter = ImgSegmMatWraper()
    # instantiate STL object
    stl_grayout = SelfTaughtLoc_Grayout(net, img_segmenter, \
                            params.min_sz_segm, topC = params.topC,\
                            alpha = params.alpha, \
                            obfuscate_bbox = params.obfuscate_bbox, \
                            function_stl = params.function_stl,\
                            padding = params.padding,\
                            single_color_space = params.single_color_space, \
                            no_obfuscation = params.no_obfuscation, \
                            mapping = mappingPASCAL)
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_output = bsddb.btopen(outputdb, 'c')
    db_keys = db_input.keys()
    if not params.use_fullimg_GT_label:
        classifier_name = 'OBFSEARCH_TOPC'
    else:
        classifier_name = 'OBFSEARCH_GT'
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        # get stuff from database entry
        img = anno.get_image()
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        # resize img to fit the size of the network
        image_resz = skimage.transform.resize(img,\
                                    (net.get_input_dim(), net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz)
        img_height, img_width = np.shape(image_resz)[0:2]
        # extract segments
        segment_lists = {}
        if not params.use_fullimg_GT_label:
            this_label = 'none'
            segment_lists[this_label] = stl_grayout.extract_greedy(image_resz)
        else:
            for GT_label in  anno.gt_objects.keys():
                logging.info('    -> Label {0}'.format(GT_label))
                segment_lists[GT_label] = \
                         stl_grayout.extract_greedy(image_resz, label=GT_label)
        anno.pred_objects[classifier_name] = {}
        for this_label in segment_lists.keys():
            if not params.use_fullimg_GT_label:
                assert this_label == 'none'
            # Convert the segmentation lists to BBoxes
            pred_bboxes_unnorm = segments_to_bboxes(segment_lists[this_label])
            # Normalize the bboxes
            pred_bboxes = []
            for j in range(np.shape(pred_bboxes_unnorm)[0]):
                pred_bboxes_unnorm[j].normalize_to_outer_box(BBox(0, 0, \
                                                    img_width, img_height))
                pred_bboxes.append(pred_bboxes_unnorm[j])
            # execute NMS, if requested
            if params.nms_execution:
                pred_bboxes = BBox.non_maxima_suppression( \
                                        pred_bboxes, params.nms_iou_threshold)
            # store results
            pred_obj = AnnotatedObject(label = this_label)
            pred_obj.bboxes = pred_bboxes
            anno.pred_objects[classifier_name][this_label] = pred_obj
            logging.info(str(anno))
            # adding the AnnotatedImage to the database
            logging.info('Adding the record to the database')
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
