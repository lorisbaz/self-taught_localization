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
import exp03

if __name__ == "__main__":
    conf = Configuration()
    params = exp03.Params()
    # experiment name
    params.exp_name = 'exp03_01'
    # number of classes to elaborate
    params.num_classes = 200
    # number of images per class to elaborate
    params.num_images_per_class = -1
    # default Configuration, image and label files
    params.conf = conf
    # we first resize each image to this size, if bigger
    params.fix_sz = 600
    # output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    # number of output chunks (the number of databases to create)
    params.num_chunks = 1000
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # visualize images (for DEBUGGING)
    params.visualize_annotated_images = False
    # specify task to debug (-1 is no-debug)
    params.task = -1;
    # RUN THE EXPERIMENT
    exp03.run_exp(params)
