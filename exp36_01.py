from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from configuration import *
from compute_statistics_exp import *
import exp36

if __name__ == "__main__":
    # load configurations and parameters
    conf = Configuration(caffe_model='alexnet')
    params = exp36.Params()
    # experiment name
    params.exp_name = 'exp36_01'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp35_01'
    # default Configuration, image and label files
    params.conf = conf
    # overlap threshold for evaluation
    params.overlap_th = 0.5
    # select top K predicted bboxes
    params.topK = 10
    params.selection_type = 'equal' # 'equal': equally distributed for each class
                                    # 'each': topK for each class
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    # list of tasks to execute
    params.task = []
    logging.info('Started')
    # RUN THE EXPERIMENT
    if 1:
        exp36.run_exp(params)
