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
from util import *

class Params:
    def __init__(self):
        pass

def pipeline(inputdb, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    # retrieve all the AnnotatedImages and images from the database
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    classifier_name = 'OBFSEARCH_GT'
    # init per_image_localization with class labels
    per_image_localization = {}
    for GT_label in conf.pascal2007_classes:
        per_image_localization[GT_label] = {}
    # loop over the images
    for image_key in db_keys:
        # get database entry
        anno = pickle.loads(db_input[image_key])
        logging.info('***** Elaborating ' + os.path.basename(anno.image_name))
        # select topK to divide for each class
        n_each_GT = {}
        if params.selection_type == 'equal':
            n_each = params.topK/float(len(anno.gt_objects.keys()))
            for GT_label in anno.gt_objects.keys():
                n_each_GT[GT_label] = int(n_each)
        elif params.selection_type == 'each':
            for GT_label in anno.gt_objects.keys():
                n_each_GT[GT_label] = params.topK
        for GT_label in anno.gt_objects.keys():
            if image_key not in per_image_localization[GT_label].keys():
                per_image_localization[GT_label][image_key] = 0
            logging.info('    -> Label {0}'.format(GT_label))
            # check if correctly localized
            for gt_bbox in anno.gt_objects[GT_label].bboxes:
                for k in range(n_each_GT[GT_label]):
                    this_bbox = anno.pred_objects[classifier_name][GT_label].bboxes[k]
                    if gt_bbox.jaccard_similarity(this_bbox)>=params.overlap_th:
                        per_image_localization[GT_label][image_key] = 1
                        break
    # write the database
    logging.info('Closing ' + inputdb)
    db_input.close()
    return per_image_localization


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
        parfun.add_task(inputdb, params)
    out_localization = parfun.run()
    logging.info('Reading state-of-the-art results')
    fid = open("./ECCV14_Wangetal_CorLoc.txt")
    CorLoc_ECCV14 = {}
    CorLoc_ECCV14_AVG = 0.0
    for line in fid.readlines():
        line_values = line.split('\t')
        CorLoc_ECCV14[line_values[0]] = float(line_values[1])
        CorLoc_ECCV14_AVG += CorLoc_ECCV14[line_values[0]]
    CorLoc_ECCV14_AVG /= float(len(params.conf.pascal2007_classes))
    fid.close()
    logging.info('Merging results and visualize')
    CorLoc = {}
    n_images = {}
    CorLoc_AVG = 0.0
    outperform = 0
    print("\t\t STL-CL \t Wang ECCV14 \t")
    for GT_label in params.conf.pascal2007_classes:
        CorLoc[GT_label] = 0.0
        n_images[GT_label] = 0
        for out_loc in out_localization:
            for image in out_loc[GT_label].keys():
                CorLoc[GT_label] += out_loc[GT_label][image]
                n_images[GT_label] += 1
        if n_images[GT_label]>0:
            print("{0} \t\t {1:2.4}% \t {2}% \t".format(GT_label, \
                                CorLoc[GT_label]/float(n_images[GT_label])*100, \
                                CorLoc_ECCV14[GT_label]))
            CorLoc_AVG += CorLoc[GT_label]/float(n_images[GT_label])
            outperform += CorLoc[GT_label]/float(n_images[GT_label])*100>\
                          CorLoc_ECCV14[GT_label]
    CorLoc_AVG /= float(len(params.conf.pascal2007_classes))
    print("AVG \t\t {0:2.4}% \t {1}% \t".format(CorLoc_AVG*100, CorLoc_ECCV14_AVG))
    print("Our method outperforms {0}/{1} classes \t".format(outperform,\
                                            len(params.conf.pascal2007_classes)))
    # Save output results
    fid = open(params.output_dir + '/CorLoc.txt', 'w')
    for GT_label in params.conf.pascal2007_classes:
        fid.write("{0}\t{1:2.4}\n".format(GT_label, str(CorLoc[GT_label]/float(n_images[GT_label])*100)))
    fid.close()
    logging.info('End of the script')
