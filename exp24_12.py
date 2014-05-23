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
    params.exp_name = 'exp24_12'
    # input
    params.exp_name_input = ['exp23_10stats_NMS_05', 'exp14_05', 'FULLIMAGE']
                            # [OBFSEARCH_GT, SELECTIVESEARCH, FULLIMAGE]
    # Filter num bboxes (only if the field is present the are used)
    params.max_num_bboxes = {'OBFSEARCH_GT': sys.maxint, \
                             'OBFSEARCH_TOPC': sys.maxint,\
                             'SELECTIVESEARCH': sys.maxint,\
                             'FULLIMAGE': sys.maxint}
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
    params.run_on_anthill = False
    # list of tasks to execute
    params.task = []
    logging.info('Started')
    # tun the pipeline
    exp24.run_exp(params)
