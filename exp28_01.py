from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from heatmap import *
from network import *
from configuration import *
from windowslider import *
import exp28

if __name__ == "__main__":
    # load configurations and parameters
    conf = Configuration()
    params = exp28.Params()
    # experiment name
    params.exp_name = 'exp28_01'
    params.exp_name_input = 'exp03_06' # take results from here
    # Sliding win params (bbox size, stride)
    params.slide_win = [(150,20), (200,25)]
                        #(50, 10), (75, 15), (100, 15), (125, 20), \
                        #(150, 20), (175, 25), (200, 25)]
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # default Configuration, image and label files
    params.conf = conf
    # select network: 'CAFFE' or 'DECAF'
    params.classifier = 'CAFFE'
    params.center_only = True
    # select top C classes used to generate the heatmaps
    params.topC = 5
    params.use_fullimg_GT_label = True # if True, params.topC ignored
    if params.use_fullimg_GT_label:
        params.topC = 0
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    # Set jobname in case the process stop or crush
    params.task = [0] # specify tasks to debug
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp28.run_exp(params)

