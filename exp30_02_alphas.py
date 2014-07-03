from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from configuration import *
from compute_statistics_exp import *
import exp30_alphas

if __name__ == "__main__":
    # load configurations and parameters
    conf = Configuration()
    params = exp30_alphas.Params()
    # experiment name
    params.exp_name = 'exp30_02_alphas'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp03_07'
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # default Configuration, image and label files
    params.conf = conf
    # select network: 'CAFFE' or 'DECAF'
    params.classifier = 'CAFFE'
    params.center_only = True
    # select top C classes used to generate the predicted bboxes
    params.topC = 5     # if 0, take the max across classes
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'
    # obfuscation search params
    params.num_of_elements_per_alpha = 5
    params.num_alphas = 3
    params.min_sz_segm = 5 # keep this low (because we resize!!)
    params.function_stl = 'similarity'
    params.obfuscate_bbox = True
    params.use_fullimg_GT_label = True # if true params.topC is not used!
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    # list of tasks (1 task -> 1 set of alphas)
    params.task = range(2)
    # list of shards to execute
    params.execute_shards = range(2) # Subset of shards to run this analysis
    logging.info('Started')
    # RUN the experiment
    exp30_alphas.run_exp(params)
