from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from heatmap import *
from configuration import *
from htmlreport import *
import exp09

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp09.Params()
    # experiment name
    params.exp_name = 'exp08_01'
    params.exp_name_input = 'exp08_01' # take results from here
    # the maximum size of an image in the html files
    params.html_max_img_size = 200
    params.visual_factor = 2 * 10e-2 # normalize output for better visualization
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
    params.job_name = None # set to None if you do not want to resume things
    params.task = None # specify task to debug      
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp09.run_exp(params)

