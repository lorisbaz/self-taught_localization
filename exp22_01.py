from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from compute_statistics_exp import *
from configuration import *
import exp22

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp22.Params()
    # experiment name
    params.exp_name = 'exp22_01'
    # take results from here
    params.exp_name_input = 'exp06_10'
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
    exp22.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    compute_statistics_exp(input_exp=params.exp_name)
