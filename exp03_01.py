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
from util import *
import exp03

if __name__ == "__main__":
    conf = Configuration()
    params = exp03.Params()
    # experiment name
    params.exp_name = 'exp03_01'
    # number of classes to elaborate
    params.num_classes = 2
    # number of images per class to elaborate
    params.num_images_per_class = 3
    # default Configuration, image and label files
    params.conf = conf
    # we first resize each image to this size, if bigger
    params.fix_sz = 600
    # the maximum size of an image in the html files
    params.html_max_img_size = 200
    # output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    # RUN THE EXPERIMENT
    exp03.run_exp(params)
