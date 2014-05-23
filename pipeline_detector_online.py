import cPickle as pickle
import bsddb
import gc
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
import vlg.util.parfun
import vlg.util.pbar

from detector import *
from featextractor import *
from heatextractor import *
import vlg.util.pbar
from util import *
from stats import *
from featprocessing import *

class PipelineDetectorParams:

    def __init__(self):
        """
        Default parameters.
        Note: all the parameters "None" are mandatory and must be filled out.
        """

        # *******************  MANDATORY PARAMETERS TO SET *******************

        # Input directories, containing the AnnotatedImages in Pickle2 format.
        # The pipeline expects a file for each <key> (i.e. <key>.pkl).
        # The AnnotatedImages might contain the features as well.
        # For differentiating the directories for training and test, see below.
        self.input_dir_train = None
        self.input_dir_test = None
        # Output directory for the pipeline. The output directory contains
        # the following elements:
        #   category_dog/ // a directory for each category
        #     iter0.pkl   // containing the PipelineDetector object of the
        #     ...         // completed iterations.
        #     iter5.pkl
        self.output_dir = None
        # Text file, containing the classes to elaborate.
        self.categories_file = None
        # Directory containing the splits. It is expected to contain
        # to text files for each class:
        #   <class>_train.txt   defining the training set
        #   <class>_test.txt    defining the test set
        # Each line of these two textfiles is in the format '<key> <+1/-1>'.
        # The suffix 'train' and 'test' can be changed, see options below.
        self.splits_dir = None

        # FeatureExtractor module to use (parameters object)
        self.feature_extractor_params = None
        # Detector module to use (parameters object)
        self.detector_params = None
        # which field of AnnotatedImage the pipeline should use for the
        # positive bboxes. We pick the bboxes of the correct label of course.
        # Format:
        # 'GT'   or
        # 'PRED:<name>'
        self.field_name_pos_bboxes = None
        # which field of AnnotatedImage the pipeline should use
        # for the bboxes used at prediction time, as well as to compose the
        # negative set. We expect the selected dictionary to have a single
        # class. Format:
        # 'PRED:<name>'  and we assume that a single category is present
        self.field_name_bboxes = None

        # *******************  OPTIONAL PARAMETERS TO SET *******************

        # experiment name. used for the job names
        self.exp_name = None
        # categories to learn (None means everything)
        self.categories_to_process = None
        # ProgressBar visualization
        self.progress_bar_params = vlg.util.pbar.ProgressBarPlusParams()
        # ParFunParams (for the categories)
        self.parfun_params_categories = vlg.util.parfun.ParFunDummyParams()
        # ParFunParams (for the training)
        self.parfun_params_training = vlg.util.parfun.ParFunDummyParams()
        # ParFunParams (for the evaluation)
        self.parfun_params_evaluation = vlg.util.parfun.ParFunDummyParams()
        # Feature preprocessing: FeatProcessorParams
        self.feature_processor_params = FeatProcessorIdentityParams()

        # name of the split for the training set
        self.split_train_name = 'train'
        # name of the split for the testing set
        self.split_test_name = 'test'

        # number of iterations to perform
        self.num_iterations = 1
        # num of negative bboxes from a positive image to add during the init
        self.num_neg_bboxes_per_pos_image_during_init = 3000
        # while selecting the neg bboxes that slightly overlap with a pos bbox
        # Params: [min_overlap, max_overlap, max_overlap_with_pos, nms_overlap]
        self.neg_bboxes_overlapping_with_pos_params = [0.0, 0.3, 0.5, 0.7]
        # thresholds to define duplicate boxes for the evaluation
        self.threshold_duplicates = 0.3
        # threshold for the confidence score for the negative examples
        self.negatives_threshold_confidence_single_image = -1.0001
        # threshold for the confidence score for the negative examples
        self.negatives_threshold_confidence_entire_set = 1.2
        # max number of positive images per category
        self.max_train_pos_images_per_category = sys.maxint
        # max number of negative images per category
        self.max_train_neg_images_per_category = sys.maxint

#==============================================================================

class PipelineImage:
    def __init__(self, key, label, fname, feature_extractor_params, \
                 field_name_pos_bboxes, field_name_bboxes, \
                 neg_bboxes_overlapping_with_pos_params):
        # check input
        assert isinstance(key, str)
        assert isinstance(label, int)
        assert isinstance(fname, str)
        assert isinstance(feature_extractor_params, FeatureExtractorParams)
        # the key of the image
        self.key = key
        # the label (+1, -1)
        self.label = label
        # the full pathname of the associated AnnotatedImage
        self.fname = fname
        # the parameters to use for the feature extractor module
        self.feature_extractor_params = feature_extractor_params
        # the bboxes (retrieved from the AI)
        self.bboxes_ = None
        # Each element of this list is a boolean  indicating whether
        # or not the bbox has been already used as a negative example.
        self.bboxes_mark_ = None
        # the confidence of the bboxes
        self.bboxes_confidence_ = None
        # the annotated image
        self.ai_ = None
        # field name to use for the pos bboxes
        # 'GT:<category>'  or
        # 'PRED:<name>:<category>'
        self.field_name_pos_bboxes = field_name_pos_bboxes
        # field name to use for the neg/pred bboxes
        # 'PRED:<name>'
        self.field_name_bboxes = field_name_bboxes
        # parameters: while selecting the neg bboxes that slightly overlap
        # with a pos bbox
        self.neg_bboxes_overlapping_with_pos_params = \
                neg_bboxes_overlapping_with_pos_params

    def get_bboxes(self):
        """
        Retrieve the list of bounding boxes to use for prediction
        (as well as for the neg set).
        The BBox objects have an extra boolean field 'mark'.
        """
        if self.bboxes_ != None:
            return self.bboxes_
        # extract the bboxes
        assert self.field_name_bboxes != None
        ai = self.get_ai()
        parts = self.field_name_bboxes.split(':')
        if parts[0] == 'PRED':
            assert len(parts)==2
            assert len(ai.pred_objects[parts[1]]) == 1
            category = ai.pred_objects[parts[1]].keys()[0]
            self.bboxes_ = ai.pred_objects[parts[1]][category].bboxes
        else:
            raise ValueError('parts[0]:{0} not recognized'.format(parts[0]))
        for idx_bb, bb in enumerate(self.bboxes_):
            # if the mark field does not exist, we create it
            if not hasattr(bb, 'mark'):
                bb.mark = False
            # if we have previsouly stored the confidences and marks,
            # we substitute them here
            if self.bboxes_mark_ != None:
                assert len(self.bboxes_)==len(self.bboxes_mark_)
                bb.mark = self.bboxes_mark_[idx_bb]
            if self.bboxes_confidence_ != None:
                assert len(self.bboxes_)==len(self.bboxes_confidence_)
                bb.confidence = self.bboxes_confidence_[idx_bb]
        if len(self.bboxes_) <= 0:
            logging.warning('Warning. The AnnotatedImage {0} '\
                'does not contain bboxes under the field {1}'\
                .format(fname, self.field_name_bboxes))
        # return
        return self.bboxes_

    def get_pos_bboxes(self):
        """
        Retrieve the positive bounding boxes, if any.
        Returns a list of Bbox objects.
        Note that len(self.bboxes_marked)==len(out)
        """
        assert self.field_name_pos_bboxes != None
        parts = self.field_name_pos_bboxes.split(':')
        ai = self.get_ai()
        if parts[0] == 'GT':
            assert len(parts)==2
            return ai.gt_objects[parts[1]].bboxes
        elif parts[0] == 'PRED':
            assert len(parts)==3
            return ai.pred_objects[parts[1]][parts[2]].bboxes
        else:
            raise ValueError('parts[0]:{0} not recognized'.format(parts[0]))

    def get_gt_bboxes(self):
        """
        Retrieve the positive bounding boxes, if any.
        Returns a list of Bbox objects.
        We use the category specified in 'field_name_pos_bboxes'
        to retrieve the correct bboxes from the GT field of AnnotatedImage.
        """
        assert self.field_name_pos_bboxes != None
        parts = self.field_name_pos_bboxes.split(':')
        ai = self.get_ai()
        bboxes = []
        if parts[0] == 'GT':
            assert len(parts)==2
            if parts[1] in ai.gt_objects:
                bboxes = ai.gt_objects[parts[1]].bboxes
        elif parts[0] == 'PRED':
            assert len(parts)==3
            if parts[2] in ai.gt_objects:
                bboxes = ai.gt_objects[parts[2]].bboxes
        else:
            raise ValueError('parts[0]:{0} not recognized'.format(parts[0]))
        return bboxes

    def get_ai(self):
        """
        Returns the associated AnnotatedImage.
        """
        # load the image from the disk, if never done it before
        if not self.ai_:
            fd = open(self.fname, 'r')
            self.ai_ = pickle.load(fd)
            fd.close()
            assert self.ai_.image_name == self.key
            # HACK. This is a pure hack to make some old-code running,
            #       when the AnnotatedImage.feature_extractor_
            #       was pickled.
            self.ai_.feature_extractor_ = None
            # register the feature extractor
            self.ai_.register_feature_extractor(self.feature_extractor_params)
        # return the AnnotatedImage
        return self.ai_

    def save_ai(self):
        """
        Dump the AnnotatedImage to the disk, overwriting the old one.
        """
        fd = open(self.fname, 'wb')
        pickle.dump(self.ai_, fd, protocol=2)
        fd.close()

    def clear_ai(self):
        """
        Clear the (eventually) loaded AnnotedImage from the memory
        """
        # clear the AI
        self.ai_ = None
        # save marks and bboxes, amd clear
        self.bboxes_mark_ = [False]*len(self.bboxes_)
        self.bboxes_confidence_ = [0.0]*len(self.bboxes_)
        assert len(self.bboxes_mark_)==len(self.bboxes_)
        assert len(self.bboxes_confidence_)==len(self.bboxes_)
        for idx_bb, bb in enumerate(self.bboxes_):
            self.bboxes_mark_[idx_bb] = bb.mark
            self.bboxes_confidence_[idx_bb] = bb.confidence
        self.bboxes_ = None
        # run the GC
        gc.collect()

    def train_elaborate_pos_example_(self, iteration,
                                     num_neg_bboxes_per_pos_image_during_init):
        """ Elaborate a positive example during the training phase.
        It marks eventual self.bboxes as negatives.
        It returns the set of positive BBox. """
        assert self.label == 1
        assert iteration == 0 # TODO: support multiple iterations
        pos_bboxes = self.get_pos_bboxes()
        if iteration == 0:
            # extract the positives, and the slightly overlapping bboxes
            # which will be used as negatives
            PipelineDetector.mark_bboxes_sligtly_overlapping_with_pos_bboxes_( \
                pos_bboxes, self.get_bboxes(), \
                num_neg_bboxes_per_pos_image_during_init, \
                self.neg_bboxes_overlapping_with_pos_params)
        else:
            # TODO. add negative examples from the positive image?
            #       For now, do nothing.
            pass
        # return
        return pos_bboxes

#==============================================================================

def pipeline_single_detector(cl, params):
    detector = PipelineDetector(cl, params)
    detector.init()
    detector.train_evaluate()
    return 0

def PipelineDetector_train_elaborate_single_image(pi, detector, iteration, \
            threshold_confidence, num_neg_bboxes_per_pos_image_during_init, \
            feature_processor):
    """ Given a PipelineImage, it evaluates the detector on all the
    bboxes, and then select only the ones that have confidence score
    above a certain threshold.
    If the detector is not given, we select all the bboxes. """
    #logging.info('Elaborating train key: {0}'.format(pi.key))
    assert (pi.label == 1) or (pi.label == -1)
    assert iteration == 0 # TODO: support multiple iterations
    # select pos and neg bboxes, building our train set
    pos_bboxes = []
    if pi.label == 1:
        # elaborate POSITIVE image.
        # Note: the method marks the negatives self.bboxes
        pos_bboxes = pi.train_elaborate_pos_example_( \
                            iteration, \
                            num_neg_bboxes_per_pos_image_during_init)
    elif pi.label == -1:
        # elaborate NEGATIVE image. we simply use all the bboxes
        for bb in pi.get_bboxes():
            bb.mark = True
    # evaluate the model learned in the previous iteration
    bboxes = pi.get_bboxes()
    if detector != None:
        for bb in bboxes:
            if not bb.mark:
                continue
            feat = pi.get_ai().extract_features(bb)
            feature_processor.process(feat)
            bb.confidence = detector.predict(feat)
            if bb.confidence < threshold_confidence:
                bb.mark = False
    # extract the features
    Xtrain = None
    Ytrain = None
    neg_bboxes = [b for b in pi.get_bboxes() if b.mark]
    n = len(pos_bboxes) + len(neg_bboxes)
    idx = 0
    for bb in pos_bboxes:
        feat = pi.get_ai().extract_features(bb)
        feature_processor.process(feat)
        if Xtrain == None:
            Xtrain, Ytrain = PipelineDetector.create_buffer_(feat.size, n)
        Xtrain[idx, :] = feat
        Ytrain[idx] = 1
        idx += 1
    for bb in neg_bboxes:
        feat = pi.get_ai().extract_features(bb)
        feature_processor.process(feat)
        if Xtrain == None:
            Xtrain, Ytrain = PipelineDetector.create_buffer_(feat.size, n)
        Xtrain[idx, :] = feat
        Ytrain[idx] = -1
        idx += 1
    # clear the AnnotatedImage
    pi.clear_ai()
    # return a tuple
    return (Xtrain, Ytrain, pi)

def PipelineDetector_evaluate_single_image(pi, detector, category, \
                                           threshold_duplicates, \
                                           feature_processor):
    # logging.info('Elaborating test key: {0}'.format(pi.key))
    # extract the features for this image
    Xtest = None
    bboxes = pi.get_bboxes()
    for idx_bb, bb in enumerate(bboxes):
        feat = pi.get_ai().extract_features(bb)
        feature_processor.process(feat)
        if Xtest == None:
            Xtest = np.empty((len(bboxes),feat.size), dtype=float)
        Xtest[idx_bb, :] = feat
    # predict the confidences
    # TODO: save the raw confidences to the disk (useful for future purposes),
    #       and return a new pi
    confidences = detector.predict(Xtest)
    assert len(confidences) == Xtest.shape[0]
    assert len(confidences) == len(bboxes)
    for idx_bb, bb in enumerate(bboxes):
        bb.confidence = confidences[idx_bb]
    # perform NMS
    pred_bboxes = BBox.non_maxima_suppression(bboxes, threshold_duplicates)
    # calculate the Stats
    gt_bboxes = pi.get_gt_bboxes()
    stats = Stats()
    stats.compute_stats(pred_bboxes, gt_bboxes)
    pi.clear_ai()
    gc.collect()
    return stats

class PipelineDetector:
    def __init__(self, category, params):
        # check the input parameters
        assert isinstance(category, str)
        assert isinstance(params, PipelineDetectorParams)
        # check that all the mandatory PipelineDetectorParams were set
        assert params.input_dir_train != None
        assert params.input_dir_test != None
        assert params.output_dir != None
        assert params.splits_dir != None
        assert params.feature_extractor_params != None
        assert params.detector_params != None
        assert params.field_name_pos_bboxes != None
        assert params.field_name_bboxes != None
        # init
        self.category = category
        self.params = params
        self.train_set = None
        self.test_set = None
        self.detector_output_dir = '{0}/{1}'.format(params.output_dir, category)
        self.iteration = 0
        self.detector = None
        self.feature_processor = None

    def init(self):
        logging.info('Initializing the detector for {0}'.format(self.category))
        # create output directory for this detector
        if os.path.exists(self.detector_output_dir) == False:
            os.makedirs(self.detector_output_dir)
        # log to both console as well as to a text file
        logging.getLogger().handlers = []
        logging.getLogger().setLevel(logging.INFO)
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(LOG_FORMATTER)
        logging.getLogger().addHandler(consoleHandler)
        fileHandler = logging.FileHandler(self.detector_output_dir + '/out.log')
        fileHandler.setFormatter(LOG_FORMATTER)
        logging.getLogger().addHandler(fileHandler)
        # read the training set
        logging.info('Read the training set files')
        fname = '{0}/{1}_{2}.txt'.format(self.params.splits_dir, self.category,\
                                         self.params.split_train_name)
        key_label_list = self.read_key_label_file_( \
                    fname, self.params.max_train_pos_images_per_category, \
                    self.params.max_train_neg_images_per_category)
        self.train_set = self.create_pipeline_images_( \
                    key_label_list, self.params, self.params.input_dir_train, \
                    self.category)
        # read the test set
        logging.info('Read the test set files')
        fname = '{0}/{1}_{2}.txt'.format(self.params.splits_dir, self.category, \
                                         self.params.split_test_name)
        key_label_list = self.read_key_label_file_(fname, sys.maxint, sys.maxint)
        self.test_set = self.create_pipeline_images_( \
                    key_label_list, self.params, self.params.input_dir_test, \
                    self.category)
        # check: make sure all the files exists
        logging.info('Checking that all the required files exist')
        error = False
        progress = vlg.util.pbar.ProgressBar.create(self.params.progress_bar_params)
        progress.set_max_val(len(self.train_set) + len(self.test_set))
        for pi in (self.train_set + self.test_set):
            progress.next()
            if not os.path.exists(pi.fname):
                error = True
                logging.info('The file {0} does not exist'.format(pi.fname))
        assert not error, 'Some required files were not found. Abort.'
        logging.info('Initialization complete')
        # create the feature processor
        self.feature_processor = FeatProcessor.create_feat_processor( \
                                    self.params.feature_processor_params)

    def train_evaluate(self):
        for iteration in range(self.params.num_iterations):
            self.iteration = iteration
            logging.info('Iteration {0}'.format(iteration))
            # check if we already trained the model for this iteration
            fname = '{0}/iter{1}.pkl'.format(self.detector_output_dir, iteration)
            if os.path.exists(fname):
                # load the current detector
                logging.info('The model for the iteration {0} already exists.'\
                             'We load it: {1}'.format(iteration, fname))
                self.load(fname)
                assert iteration == self.iteration
            else:
                # train the detector and save the model
                logging.info('Training the  model for the iteration {0}.'\
                             .format(iteration))
                self.train()
                logging.info('Saving the model to {0}'.format(fname))
                self.save(fname)
            # check if we have already evaluated the model for this iteration
            fname = '{0}/iter_stats{1}.pkl'.format( \
                    self.detector_output_dir, iteration)
            fname_mat = '{0}/iter_stats{1}.mat'.format( \
                    self.detector_output_dir, iteration)
            if os.path.exists(fname):
                logging.info('The stats file for iteration {0} already '\
                             'exists: {1}'.format(iteration, fname))
            else:
                logging.info('Evaluation the model of iteration {0}'.format( \
                             iteration))
                stats = self.evaluate()
                logging.info('AP: {0}'.format(stats.average_prec))
                dump_obj_to_file_using_pickle(stats, fname, 'binary')
                stats.save_mat(fname_mat)

    def train(self):
        """ Train an iteration of the detector """
        logging.info('Training')
        self.detector = None
        Xtrain = None
        Ytrain = None
        num_negatives_added = 0
        progress = vlg.util.pbar.ProgressBar.create(self.params.progress_bar_params)
        progress.set_max_val(len(self.train_set))
        for idx_pi, pi in enumerate(self.train_set):
            progress.next()
            logging.info('Elaborating key {0}'.format(pi.key))
            # extract the training set for this example
            out = PipelineDetector_train_elaborate_single_image( \
                    pi, self.detector, self.iteration, \
                    self.params.negatives_threshold_confidence_single_image, \
                    self.params.num_neg_bboxes_per_pos_image_during_init, \
                    self.feature_processor)
            (Xtrain_pi, Ytrain_pi, pi_out) = out
            # put all the features together in a single matrix
            if Xtrain == None:
                Xtrain = Xtrain_pi
                Ytrain = Ytrain_pi
            else:
                if Xtrain_pi == None:
                    logging.info('We skip the training because this image did '\
                                 'not provide any training example')
                    continue
                else:
                    Xtrain = np.vstack([Xtrain, Xtrain_pi])
                    Ytrain = np.concatenate([Ytrain, Ytrain_pi])
            num_examples = len(Ytrain)
            num_pos_examples = len([y for y in Ytrain if y==1])
            num_neg_examples = len([y for y in Ytrain if y==-1])
            num_negatives_added += len([y for y in Ytrain_pi if y==-1])
            logging.info('The training set has {0} positive and {1} negative '\
                     'examples'.format(num_pos_examples, num_neg_examples))
            assert Xtrain.shape[0] == num_examples
            assert Ytrain.size == num_examples
            # if the training set has no positives, we skip the training
            if num_pos_examples==0 or num_neg_examples==0:
                logging.info('We skip the training because one of the classes '\
                             'is not represented')
                continue
            # if necessary, we update the detector
            if (idx_pi == 0) \
                    or (idx_pi == len(self.train_set)-1) \
                    or (num_negatives_added >= 2000):
                logging.info('Train the detector')
                num_negatives_added = 0
                # training
                if self.detector == None:
                    self.detector = Detector.create_detector( \
                                        self.params.detector_params)
                self.detector.train(Xtrain, Ytrain)
                # we remove the easy examples
                scores = self.detector.predict(Xtrain)
                thresh = self.params.negatives_threshold_confidence_entire_set
                examples_to_keep = [(Ytrain[i]==1) or (Ytrain[i]==-1 and s>=thresh) \
                                    for i, s in enumerate(scores)]
                examples_to_keep = np.array(examples_to_keep)
                Xtrain = Xtrain[examples_to_keep, :]
                Ytrain = Ytrain[examples_to_keep]
        progress.finish()

    def evaluate(self):
        """ calculate the stats for the current model """
        # calculate stats for the individual images
        parfun = vlg.util.parfun.ParFun.create(self.params.parfun_params_evaluation)
        parfun.set_fun(PipelineDetector_evaluate_single_image)
        for pi in self.test_set:
            parfun.add_task(pi, self.detector, self.category, \
                            self.params.threshold_duplicates, \
                            self.feature_processor)
        stats_all = parfun.run()
        assert len(stats_all)==len(self.test_set)
        # aggregate the stats for this detector
        stats, hist_overlap = Stats.aggregate_results(stats_all)
        return stats

    def load(self, fname):
        """ Load from a Pickled file, and substitute the current fields """
        fd = open(fname, 'r')
        pd = pickle.load(fd)
        fd.close()
        assert self.category == pd.category
        self.params = pd.params
        self.train_set = pd.train_set
        self.test_set = pd.test_set
        assert self.detector_output_dir == pd.detector_output_dir
        self.iteration = pd.iteration
        self.detector = pd.detector

    def save(self, fname):
        """ Pickle and save the current object to a file """
        dump_obj_to_file_using_pickle(self, fname, 'binary')

    @staticmethod
    def create_buffer_(num_dims, buffer_size):
        Xtrain = np.zeros(shape=(buffer_size,num_dims), dtype=float)
        Ytrain = np.zeros(shape=(buffer_size), dtype=int)
        return Xtrain, Ytrain

    @staticmethod
    def mark_bboxes_sligtly_overlapping_with_pos_bboxes_( \
                            pos_bboxes, bboxes, max_num_bboxes, \
                            params = [0.2, 0.5, 0.5, 0.7]):
        """
        Mark the bboxes that sligtly overlap with the pos bboxes.
        If there are too many bboxes, we
        randomly-chosen subset of 'max_num_bboxes' bboxes.
        Input: pos_bboxes: is a list of BBox objects.
               bboxes: is a list of BBox (with the 'mark' field)
        Output: Nothing.
        """
        # check input
        assert isinstance(pos_bboxes, list)
        assert isinstance(bboxes, list)
        for bb in pos_bboxes:
            assert isinstance(bb, BBox)
        for bb in bboxes:
            assert isinstance(bb, BBox)
            assert bb.mark == False
        assert max_num_bboxes > 0
        assert isinstance(params, list)
        assert len(params) == 4
        min_overlap = params[0]
        max_overlap = params[1]
        max_overlap_with_pos = params[2]
        nms_overlap = params[3]
        out = []
        # select the bboxes that have an overlap between 0.2 and 0.5
        # with any positive
        for bb in bboxes:
            for pos_bb in pos_bboxes:
                overlap = bb.jaccard_similarity(pos_bb)
                if (overlap >= min_overlap) and (overlap <= max_overlap):
                    out.append(bb)
                    break
        # remove the bboxes that overlap too much with a positive
        # (this might happen when there are >1 pos objs per image)
        out2 = []
        for bb in out:
            remove_bb = False
            for pos_bb in pos_bboxes:
                overlap = bb.jaccard_similarity(pos_bb)
                if overlap > max_overlap_with_pos:
                    remove_bb = True
                    break
            if not remove_bb:
                out2.append(bb)
        out = out2
        # randomly shuffle
        out = [out[i] for i in util.randperm_deterministic(len(out))]
        # remove near-duplicates
        out2 = []
        while len(out) > 0:
            bb = out.pop()
            out2.append(bb)
            out = [bb2 for bb2 in out \
                   if bb.jaccard_similarity(bb2) <= nms_overlap]
        out = out2
        # mark the bboxes to keep
        for i in range(min(len(out), max_num_bboxes)):
            out[i].mark = True

    @staticmethod
    def create_pipeline_images_(key_label_list, params, input_dir, category):
        """
        Input: list of (<key>, <+1/-1>), and PipelineDetectorParams.
        Output: list of PipelineImage
        """
        assert isinstance(key_label_list, list)
        assert isinstance(params, PipelineDetectorParams)
        out = []
        progress = vlg.util.pbar.ProgressBar.create(params.progress_bar_params)
        progress.set_max_val(len(key_label_list))
        for idx, key_label in enumerate(key_label_list):
            key, label = key_label
            fname = '{0}/{1}.pkl'.format(input_dir, key)
            #logging.info('Create PipelineImage for {0} ({1}/{2})'.format( \
            #             fname, idx+1, len(key_label_list)))
            progress.update(idx+1)
            # create the PipelineImage
            pos_bboxes_field_name = '{0}:{1}'.format( \
                            params.field_name_pos_bboxes, category)
            bboxes_field_name = params.field_name_bboxes
            pi = PipelineImage( \
                    key, label, fname, \
                    params.feature_extractor_params, \
                    pos_bboxes_field_name, bboxes_field_name, \
                    params.neg_bboxes_overlapping_with_pos_params)
            # append the PipelineImage
            out.append(pi)
        return out

    @staticmethod
    def read_key_label_file_(fname, max_pos_examples, max_neg_examples):
        """
        Read a text file each line being '<key> <+1/-1>'.
        We select randomly at most max_pos_examples and max_neg_examples.
        Returns a list of tuples (<key>, <+1/-1>) where key is a string.
        The list is randomly shuffled.
        """
        assert isinstance(fname, str)
        assert max_pos_examples >= 0
        assert max_neg_examples >= 0
        out = []
        # read the file
        pos_set = []
        neg_set = []
        fd = open(fname, 'r')
        for line in fd:
            elems = line.strip().split()
            assert len(elems)==2
            key, label = elems
            if int(label) == 1:
                pos_set.append(key)
            elif int(label) == -1:
                neg_set.append(key)
            else:
                raise ValueError('The label {0} is not recognized'.format(label))
        fd.close()
        # subsample randomly the set
        idx_pos = randperm_deterministic(len(pos_set))
        idx_pos = idx_pos[0:min(len(idx_pos), max_pos_examples)]
        idx_neg = randperm_deterministic(len(neg_set))
        idx_neg = idx_neg[0:min(len(idx_neg), max_neg_examples)]
        for i in idx_pos:
            out.append( (pos_set[i], 1) )
        for i in idx_neg:
            out.append( (neg_set[i], -1) )
        # shuffle randomly the set
        idxperm = randperm_deterministic(len(out))
        return [out[i] for i in idxperm]

    @staticmethod
    def train_evaluate_detectors(params):
        """
        Train a set of detectors.
        """
        # create output directory
        if os.path.exists(params.output_dir) == False:
            os.makedirs(params.output_dir)
        # read the list of classes to elaborate
        classes = []
        fd = open(params.categories_file, 'r')
        for line in fd:
            classes.append(line.strip())
        fd.close()
        logging.info('Loaded {0} classes'.format(len(classes)))
        # select the categories to process
        if params.categories_to_process:
            classes = [classes[i] for i in params.categories_to_process]
        # run the pipeline
        if params.exp_name != None:
            params.parfun_params_categories.job_name = \
                    'Job{0}'.format(params.exp_name).replace('exp','')
        parfun = vlg.util.parfun.ParFun.create(params.parfun_params_categories)
        parfun.set_fun(pipeline_single_detector)
        for cl in classes:
            parfun.add_task(cl, params)
        out = parfun.run()
        for i, val in enumerate(out):
            if val != 0:
                logging.info('Task {0} didn''t exit properly'.format(i))
        logging.info('End of the script')
