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
import exp23

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp23.Params()
    # experiment name
    params.exp_name = 'exp23_01'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp03_04'
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # default Configuration, image and label files
    params.conf = conf
    # select network: 'CAFFE' or 'DECAF'
    params.classifier = 'CAFFE'
    params.center_only = True
    # select top C classes used to generate the predicted bboxes
    params.topC = 0     # if 0, take the max across classes
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'     
    # selective search version
    params.ss_version = 'fast'
    params.min_sz_segm = 5 # keep this low (because we resize!!)
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
    exp23.run_exp(params)
    # RUN THE STATISTICS PIPELINE
    compute_statistics_exp(input_exp=params.exp_name)
