from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from configuration import *
from compute_statistics_exp import *
import exp33

if __name__ == "__main__":
    # load configurations and parameters
    conf = Configuration()
    params = exp33.Params()
    # experiment name
    params.exp_name = 'exp33_01'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp19_01'
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # default Configuration, image and label files
    params.conf = conf
    # select network: 'CAFFE' or 'DECAF'
    params.classifier = 'CAFFE'
    params.center_only = True
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # list of tasks to execute
    params.task = range(100)
    logging.info('Started')
    # RUN THE EXPERIMENT
    if 1:
        exp33.run_exp(params)
