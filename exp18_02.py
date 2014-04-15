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
import exp18

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp18.Params()
    # experiment name
    params.exp_name = 'exp18_02'
    # take results from here
    params.exp_name_input = 'exp16_06'
    # extract heat heatmaps from AVG heatmap and INDIVIDUAL heats
    params.extract_bbox_from_avg_heatmap = True
    params.extract_bbox_from_individual_heatmaps = True
    # Want to use the top C classes (max C = 5; if = 0 use the GT!)
    params.top_C_classes = 5
    # Sliding window over heatmap parameters (width, height, stridex, stridey)
    params.sliding_win = [(50, 50, 10, 10), (75, 50, 10, 10), \
                          (55, 75, 10, 10), (75, 75, 15, 15), \
                          (75, 100, 15, 15), (100, 75, 15, 15), \
                          (100, 100, 20, 20), (150, 100, 20, 20), \
                          (100, 150, 20, 20), (150, 150, 20, 20), \
                          (150, 175, 20, 20), (175, 150, 20, 20), \
                          (175, 175, 25, 25), (200, 175, 25, 25), \
                          (175, 200, 25, 25), (200, 200, 25, 25)]
    params.area_normalization = True 
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
    if 1:
        exp18.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    if 0:
        compute_statistics_exp(input_exp=params.exp_name)
    # RUN THE STATISTICS PIPELINE WITH NMS
    if 1:
#        # NMS=0.3
#        params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_03')
#        params_stats.nms_execution = True
#        params_stats.nms_iou_threshold = 0.3
#        compute_statistics_exp(input_exp=params.exp_name, params=params_stats)
        # NMS=0.5
        params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_05')
        params_stats.nms_execution = True
        params_stats.nms_iou_threshold = 0.5
        compute_statistics_exp(input_exp=params.exp_name, params=params_stats)
        # NMS=0.9
        params_stats = ComputeStatParams(params.exp_name, 'stats_NMS_09')
        params_stats.nms_execution = True
        params_stats.nms_iou_threshold = 0.9
        compute_statistics_exp(input_exp=params.exp_name, params=params_stats)
