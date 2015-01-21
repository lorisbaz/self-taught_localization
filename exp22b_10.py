from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from compute_statistics_exp import *
from configuration import *
import exp22b

if __name__ == "__main__":
    # load configurations and parameters
    conf = Configuration(caffe_model='alexnet')
    params = exp22b.Params()
    # experiment name
    params.exp_name = 'exp22b_10'
    # take results from here
    params.exp_name_input = 'exp14_06'
    # select classifier
    params.classifier = 'CAFFE'
    params.center_only = True
    # default Configuration, image and label files
    params.conf = conf
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # Set jobname in case the process stop or crush
    params.task = []
    logging.info('Started')
    # RUN THE EXPERIMENT
    if 1:
        exp22b.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    if 1:
        compute_statistics_exp(input_exp=params.exp_name)
