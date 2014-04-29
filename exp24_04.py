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
    params.exp_name = 'exp24_04'
    # input
    params.exp_name_input = 'exp14_08'
    # export parameters
    params.name_pred_objects = 'SELECTIVESEARCH' # THIS IS ACTUALLY OBF SEARCH!
    params.max_num_bboxes = -1
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # feature layer
    params.feature_extractor_params = load_obj_from_file_using_pickle( \
                                       'featextractor_specs/000.pkl')
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
    # tun the pipeline
    exp24.run_exp(params)
