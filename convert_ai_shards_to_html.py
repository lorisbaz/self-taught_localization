import cPickle as pickle
import bsddb
import glob
import logging
import numpy as np
import os
import os.path
import sys
import scipy.misc
import skimage.io
from vlg.util.parfun import *

from annotatedimage import *
from bbox import *
from bboxextractor import *
from heatmap import *
from configuration import *
from htmlreport import *
from util import *


#shards = range(15, 18, 1)  # bird
shards = range(50, 52, 1)   # lizzard

#exp_name = 'exp23_11'
exp_name = 'exp14_07'

nms_execution = True
nms_iou_threshold = 0.5
topK = 1

def convert_ai_shard_to_html(inputdb, outputhtml):
    htmlres = HtmlReport()
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    db_keys = db_input.keys()
    # loop over the images
    for image_key in db_keys:
        logging.info('Elaborating {0}'.format(image_key))
        # get database entry
        ai = pickle.loads(db_input[image_key])
        for classifier in ai.pred_objects:
            for label in ai.pred_objects[classifier]:
                pred_bboxes = ai.pred_objects[classifier][label].bboxes
                if nms_execution:
                    pred_bboxes = BBox.non_maxima_suppression( \
                                         pred_bboxes, nms_iou_threshold)
                pred_bboxes = pred_bboxes[0:topK]
                ai.pred_objects[classifier][label].bboxes = pred_bboxes
        # visualize the annotation to a HTML row
        htmlres.add_annotated_image_embedded(ai, \
                        img_max_size=600)
        htmlres.add_newline()
    # write the HTML
    logging.info('Writing ' + outputhtml)
    htmlres.save(outputhtml)

if __name__ == '__main__':
    conf = Configuration()
    exp_dir = conf.experiments_output_directory + '/' + exp_name
    for i in shards:
        shard_id = '%05d'%i
        inputdb = exp_dir + '/' + shard_id + '.db'
        outputhtml = exp_dir  + '/' + shard_id + '.html'
        convert_ai_shard_to_html(inputdb, outputhtml)
