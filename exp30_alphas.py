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
import shutil
from vlg.util.parfun import *

from annotatedimage import *
from bbox import *
from imgsegmentation import *
from configuration import *
from util import *
from network import *
from self_taught_loc import *
from compute_statistics_exp import *

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
        # Number of alphas to use for creating the simplex (sampling)
        self.num_of_elements_per_alpha = 10
        # If cnnfeature are used, we can include some padding to the
        # bbox where feature are extracted. value in [0.0, 1.0]
        self.padding = 0.0
        # Select a single parametrization of Feltz alg (i.e., single color space
        # and k)
        self.single_color_space = False

def pipeline(input_dir, output_dir, alphas, idx_process, params):
    # Instantiate some objects, and open the database
    conf = params.conf
    if params.select_subset_overlap_ilsvrc2012_ilsvrc2013:
        # Retrieve wnids (used to rule out GTs)
        locids, wnids_my_subset = \
                    get_wnids(conf.ilsvrc2013_classid_wnid_words_overlap)
    else:
        wnids_my_subset = []
    if params.classifier=='CAFFE':
        netParams = NetworkCaffeParams(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    elif params.classifier=='DECAF':
        netParams = NetworkDecafParams(conf.ilsvrc2012_decaf_model_spec, \
                           conf.ilsvrc2012_decaf_model, \
                           conf.ilsvrc2012_classid_wnid_words, \
                           center_only = params.center_only,\
                           wnid_subset = wnids_my_subset)
    net = Network.create_network(netParams)
    # choose segmentation method (Matlab wrapper Felz through SS)
    img_segmenter = ImgSegmMatWraper()
    # instantiate STL object
    stl_grayout = SelfTaughtLoc_Grayout(net, img_segmenter, \
                            params.min_sz_segm, topC = params.topC,\
                            alpha = alphas, \
                            obfuscate_bbox = params.obfuscate_bbox, \
                            function_stl = params.function_stl,\
                            padding = params.padding,\
                            single_color_space = params.single_color_space)
    # create folder
    if os.path.exists(output_dir + '/%05d'%idx_process) == False:
        os.makedirs(output_dir + '/%05d'%idx_process)
    # perform exp for all the shards
    if not params.only_evaluation:
        for shard in params.execute_shards:
            inputdb = input_dir + '/%05d'%shard + '.db'
            outputdb = output_dir + '/%05d'%idx_process  + '/%05d'%shard + '.db'
            try: # if DB exists, check integrity
                db_output = bsddb.btopen(outputdb, 'r')
                db_output.close()
                logging.info('--- DB successfully opened.'+\
                             ' We do not overwrite {0}'.format(outputdb))
            except:
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

    # Perform NMS = 0.5
    params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_05')
    params_stats.nms_execution = True
    params_stats.nms_iou_threshold = 0.5
    params_stats.run_on_anthill = False
    params_stats.calculate_histogram = False
    params_stats.task = params.execute_shards
    params_stats.input_dir = output_dir + '/%05d'%idx_process
    if os.path.exists(output_dir + 'stats_NMS_05') == False:
        os.makedirs(output_dir + 'stats_NMS_05')
    params_stats.output_dir = output_dir + 'stats_NMS_05' + '/%05d'%idx_process
    compute_statistics_exp(input_exp=params.exp_name, params=params_stats)

    ## clean results
    #for shard in params.execute_shards:
    #    shutil.rmtree(params_stats.input_dir +  '/%05d'%shard + '.db')
    #    shutil.rmtree(params_stats.output_dir + '/%05d'%shard + '.db')

    return 0

def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # Create the alpha grid
    alpha = []
    for i in range(params.num_alphas-1):
        alpha.append(np.linspace(0, 1, params.num_of_elements_per_alpha))
        if i == 0:
            list_var_alpha = 'alpha[' + str(i) + ']'
        else:
            list_var_alpha = list_var_alpha + ' ,alpha[' + str(i) + ']'
    alphas = eval('np.meshgrid(' + list_var_alpha + ')')
    alphas = np.reshape(alphas, (params.num_alphas-1, \
                        params.num_of_elements_per_alpha**(params.num_alphas-1)))
    # compute the last as 1-sum of the others
    alpha_last = 1 - np.sum(alphas, axis=0)
    # keep the alphas in the simplex
    keep_alphas = alpha_last >= 0.0
    alphas = np.append(alphas[:, keep_alphas], \
                    np.expand_dims(alpha_last[:, keep_alphas], axis=0), axis=0)
    # save the alphas to mat
    scipy.io.savemat(params.output_dir + '/list_of_alphas.mat', \
                    {'alphas': alphas})
    # # Decomment if you want to visualize the simplex points
    # from mpl_toolkits.mplot3d import Axes3D
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(alphas[0,:], alphas[1,:], alphas[2,:])
    # plt.show()
    # run the pipeline
    n_tasks = np.shape(alphas)[1]
    parfun = None
    if params.run_on_anthill:
        jobname = 'Job{0}'.format(params.exp_name).replace('exp','')
        parfun = ParFunAnthill(pipeline, time_requested=48, \
            job_name=jobname)
    else:
        parfun = ParFunDummy(pipeline)
    if len(params.task) == 0:
        idx_to_process = range(n_tasks)
    else:
        idx_to_process = params.task
    for i in idx_to_process:
        inputdb = params.input_dir
        outputdb = params.output_dir
        parfun.add_task(inputdb, outputdb, alphas[:, i], i, params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
