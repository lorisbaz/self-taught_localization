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
import pipeline_export_AnnotatedImage_to_text

if __name__ == "__main__":
    # default parameters
    params = pipeline_export_AnnotatedImage_to_text.ExportAnnoImageParams()
    conf = Configuration()
    params.conf = conf
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp23_09stats_NMS_05'
    params.generate_htmls = True
    params.html_max_img_size = 300
    # output textfile
    params.output_textfile = 'pred_top3_bboxes.txt'
    # export parameters
    params.name_pred_objects = 'SELECTIVESEARCH'
    params.max_num_bboxes = 3
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # list of tasks to execute
    params.task = []
    logging.info('Started')
    pipeline_export_AnnotatedImage_to_text.pipeline(params)
