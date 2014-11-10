from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from configuration import *
from compute_statistics_exp import *
import exp30_speed

if __name__ == "__main__":
    # load configurations and parameters
    conf = Configuration()
    params = exp30_speed.Params()
    # experiment name
    params.exp_name = 'exp30_07_speed'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp19_01'
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
    params.min_sz_segm = 5 # keep this low (because we resize!!)
    params.alpha = 1/4.0*np.ones((4,))
    params.function_stl = 'similarity+cnnfeature'
    params.obfuscate_bbox = True
    params.use_fullimg_GT_label = False # if true params.topC is not used!
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # list of tasks to execute
    params.task = range(30)
    logging.info('Started')
    # RUN THE EXPERIMENT
    if 1:
        exp30_speed.run_exp(params)