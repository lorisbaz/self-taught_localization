from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from heatmap import *
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from compute_statistics_exp import *
from htmlreport import *
import exp14

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp14.Params()
    # experiment name
    params.exp_name = 'exp14_02'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp03_04'
    # selective search version
    params.ss_version = 'fast'
    # default Configuration, image and label files
    params.conf = conf
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # list of tasks to execute
    params.task = []
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp14.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    compute_statistics_exp(input_exp=params.exp_name)
