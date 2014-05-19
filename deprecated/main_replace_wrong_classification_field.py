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
from configuration import *
from util import *
import replace_wrong_classification_field

if __name__=="__main__":
    conf = Configuration()
    params = replace_wrong_classification_field.Params()
    params.exp_name_input = str(sys.argv[1])
    if len(sys.argv)>3:
        logging.warning('More that 2 input argument. Ignoring the others.')
    params.field_to_be_replaced = 'SELECTIVESEARCH'
    params.replance_with_this = str(sys.argv[2]) # OBFSEARCH_TOPC, OBFSEARCH_GT
    params.exp_name_output = params.exp_name_input + '_CORRECT'
    params.input_dir = conf.experiments_output_directory \
                                        + '/' + params.exp_name_input
    params.output_dir = conf.experiments_output_directory \
                                        + '/' + params.exp_name_output
    logging.info('Elaborating: {0} '.format(params.input_dir))
    params.run_on_anthill = True
    params.task = []
    replace_wrong_classification_field.run_exp(params)

