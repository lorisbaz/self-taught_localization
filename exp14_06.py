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
    params.exp_name = 'exp14_06'
    # input (GT AnnotatatedImages for ILSVRC-val-1000)
    params.exp_name_input = 'exp03_05'
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
    if 0:
        exp14.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    if 1:
        compute_statistics_exp(input_exp=params.exp_name)    
    # RUN THE STATISTICS PIPELINE WITH NMS
    if 1:
        # NMS=0.3
        params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_03')
        params_stats.nms_execution = True
        params_stats.nms_iou_threshold = 0.3
        compute_statistics_exp(input_exp=params.exp_name, params=params_stats)
    if 1:
        # NMS=0.5
        params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_05')
        params_stats.nms_execution = True
        params_stats.nms_iou_threshold = 0.5
        compute_statistics_exp(input_exp=params.exp_name, params=params_stats)
    if 1:
        # NMS=0.9
        params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_09')
        params_stats.nms_execution = True
        params_stats.nms_iou_threshold = 0.9
        compute_statistics_exp(input_exp=params.exp_name, params=params_stats)

