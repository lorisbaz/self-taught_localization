from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from heatmap import *
from network import *
from configuration import *
from heatextractor import *
from htmlreport import *
import exp10

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp10.Params()
    # experiment name
    params.exp_name = 'exp10_01'
    params.exp_name_input = 'exp03_01' # take results from here
    # Gray box params (bbox size, stride)
    params.gray_par = [(32, 25), (48, 25), (64, 25), (80, 25), (96, 25)]
    # default Configuration, image and label files
    params.conf = conf
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'
    # normalize the confidence by area?
    params.heatextractor_area_normalization = True
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # Set jobname in case the process stop or crush
    params.job_name = None # set to None if you do not want to resume things
    params.task = None # specify task to debug
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp10.run_exp(params)

