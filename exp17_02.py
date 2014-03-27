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
from htmlreport import *
import exp17

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp17.Params()
    # experiment name
    params.exp_name = 'exp17_02'
    # input dataset (PASCAL2007-test)
    params.exp_name_input = 'exp19_01'
    # we do NOT include the GT heatmap (PASCAL has many "imagenet" labels)
    params.include_gt_label_heatmap = False
    # Sliding win params (bbox size, stride)
    params.gray_par = [(50, 10), (75, 15), (100, 15), (125, 20), \
                       (150, 20), (175, 25), (200, 25)]
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # default Configuration, image and label files
    params.conf = conf
    # select network: 'CAFFE' or 'DECAF'
    params.classifier = 'CAFFE'
    params.center_only = True
    # select top C classes used to generate the heatmaps
    params.topC = 5
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'
    # normalize the confidence by area?
    params.heatextractor_area_normalization = True
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # specify which shards to execute
    params.task = []
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp17.run_exp(params)

