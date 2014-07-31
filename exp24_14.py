from util import *
import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from network import *
from configuration import *
import exp24

if __name__ == "__main__":
    # default parameters
    params = exp24.Params()
    conf = Configuration()
    params.conf = conf
    # experiment name
    params.exp_name = 'exp24_14'
    # input [OBFSEARCH_GT, SELECTIVESEARCH, FULLIMAGE]
    params.exp_name_input = ['exp30_03stats_NMS_05', 'exp14_07']
    params.full_image_boxes = False
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # feature layer: we are loading the Girshick caffe model (slightly
    # different from the imagenet one)
    params.feature_extractor_params = load_obj_from_file_using_pickle( \
                                       'featextractor_specs/001.pkl')
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = []
    for names in params.exp_name_input:
        params.input_dir.append(conf.experiments_output_directory + '/' + names)
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # list of tasks to execute
    params.task = []
    logging.info('Started')
    # tun the pipeline
    exp24.run_exp(params)
