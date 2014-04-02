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
import exp06

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp06.Params()
    # experiment name
    params.exp_name = 'exp06_11'
    # take results from here
    params.exp_name_input = 'exp17_01'
    # extract heat heatmaps from AVG heatmap and INDIVIDUAL heats
    params.extract_bbox_from_avg_heatmap = True
    params.extract_bbox_from_individual_heatmaps = True
    # Want to use the top C classes (max C = 5; if = 0 use the GT!)
    params.top_C_classes = 5
    # Bounding box  parameters
    params.min_bbox_size = 0.02
    params.grab_cut_rounds = 30
    params.consider_pr_fg = True
    # default Configuration, image and label files
    params.conf = conf
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # max size of the HTML images
    params.html_max_img_size = 300
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # Set jobname in case the process stop or crush
    params.task = []
    # specify task to debug 
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp06.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    compute_statistics_exp(input_exp=params.exp_name)
