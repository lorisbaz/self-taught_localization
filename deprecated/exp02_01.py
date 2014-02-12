import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from heatmap import *
from network import *
from configuration import *
#from imgsegmentation import *
from heatextractor import *
from htmlreport import *
from util import *
import exp02

if __name__ == "__main__":
    conf = Configuration()
    params = exp02.Params()    
    # experiment name
    params.exp_name = 'exp02_01'
    # number of classes to elaborate
    params.num_classes = 10
    # number of images per class to elaborate
    params.num_images_per_class = 20 
    # default Configuration, image and label files
    params.conf = conf
    params.images_file = conf.ilsvrc2012_val_images
    params.labels_file = conf.ilsvrc2012_val_labels
    # Felzenswalb segmentation params: (scale, sigma, min)
    params.box_params = [(20, 10), \
                  	 (40, 10), \
                  	 (60, 10), \
			 (80, 10)]
    # we first resize each image to this size, if bigger 
    params.fix_sz = 300
    # the maximum size of an image in the html files
    params.html_max_img_size = 200
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'
    # normalize the confidence by area?
    params.heatextractor_area_normalization = False
    # output directory
    params.output_dir = conf.experiments_output_directory \
			+ '/' + params.exp_name
    # parallelize the script on Anthill?
    params.run_on_anthill = False
    # visualize heatmaps?
    params.visualize_heatmaps = True
    # RUN THE EXPERIMENT
    exp02.run_exp(params)
