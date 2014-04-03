from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from configuration import *
import exp08

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp08.Params()
    # experiment name
    params.exp_name = 'exp08_01'
    params.exp_name_input = 'exp06_01' # take results from here
    # default Configuration, image and label files
    params.conf = conf
    # Intersection over Union threshold
    params.IoU_threshold = 0.5
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # parallelize the script on Anthill?
    params.run_on_anthill = True 
    # specify task to debug
    params.task = None
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp08.run_exp(params)
