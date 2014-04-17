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
import exp19

if __name__ == "__main__":
    conf = Configuration()
    params = exp19.Params()
    # experiment name
    params.exp_name = 'exp19_02'
    # default Configuration
    params.conf = conf
    # sets to include
    params.sets = ['trainval']
    # include difficult objects
    params.include_difficult_objects = False
    # output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    # number of output chunks (the number of databases to create)
    params.num_chunks = 1000
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # specify task to debug ([] process everything)
    params.task = []
    # RUN THE EXPERIMENT
    exp19.run_exp(params)

