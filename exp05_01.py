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
import exp05

if __name__ == "__main__":
    # load configurations and parameters  
    conf = Configuration()
    params = exp05.Params()
    # experiment name
    params.exp_name = 'exp05_01'
    # number of classes to elaborate
    params.num_classes = 20
    # number of images per class to elaborate
    params.num_images_per_class = 10
    # default Configuration, image and label files
    params.conf = conf
    # we first resize each image to this size, if bigger
    params.fix_sz = 600
    # the maximum size of an image in the html files
    params.html_max_img_size = 200
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'
    # normalize the confidence by area?
    params.heatextractor_area_normalization = True
    # output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    # number of output chunks (the number of databases to create)
    params.num_chunks = 1000
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    # visualize images (for DEBUGGING)
    params.visualize_annotated_images = True 
    # segmentation masks?
    params.visualize_segmentation_masks = True
    # visualize heatmaps?
    params.visualize_heatmaps = True
    params.visual_factor = 10e-2 # normalize output for
				 # better visualization
    logging.info('Started')
    # RUN THE EXPERIMENT
    exp05.run_exp(params)

