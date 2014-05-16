import cPickle as pickle
import bsddb
import glob
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import scipy.io
import skimage.io
import tempfile
from vlg.util.parfun import *

from annotatedimage import *
from bbox import *
from configuration import *
from util import *
import check_integrity_pred_objects

if __name__=="__main__":
    conf = Configuration()
    params = check_integrity_pred_objects.Params()
    params.exp_name_input = 'exp23_08'
    params.input_dir = conf.experiments_output_directory \
                                        + '/' + params.exp_name_input
    params.run_on_anthill = True
    params.task = []
    check_integrity_pred_objects.run_exp(params)

