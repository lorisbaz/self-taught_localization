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
    params.exp_name = 'exp08_17'
    # take results from here
    params.exp_name_input = 'exp06_12'
    # default Configuration, image and label files
    params.conf = conf
    # Intersection over Union threshold
    params.IoU_threshold = 0.5
    # create also some statistics using a variable number of predictions/image
    params.stats_using_num_pred_bboxes_image = range(1,16)
    #params.stats_using_num_pred_bboxes_image.extend([20, 30, 50]) 
    # max num of subwindows generated per image (if 0, take them all)
    params.max_subwin = 6e3 
    # delete the pred_objects from the AnnotatedImages
    params.delete_pred_objects = True
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/ '+ params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    params.run_stat_pipeline = True
    # Set jobname in case the process stop or crush
    params.task = []
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp08.run_exp(params)

