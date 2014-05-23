from util import *
import numpy as np
import os
import cPickle as pickle
import os.path
import bsddb
import glob
from vlg.util.parfun import *

from configuration import *

class Params:
    def __init__(self):
        pass

def run_generate_pos(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # create support files
    n_chunks = len(glob.glob(params.input_dir + '/*.db'))
    list_files = []
    positive_set = {}
    for i in range(n_chunks):
        inputdb = params.input_dir + '/%05d'%i + '.db'
        logging.info('***** Processing database {0}'.format(inputdb))
        db_input = bsddb.btopen(inputdb, 'r')
        db_keys = db_input.keys()
        for image_key in db_keys:
            # get database entry
            anno = pickle.loads(db_input[image_key])
            logging.info('(+) Positive Image {0}'.format(anno.image_name))
            # put the image name in the correct file
            for label in anno.gt_objects.keys():
                filename = params.output_dir + '/' + \
                                label + '_' + params.append_name + '.txt'
                # open file in append mode
                if filename in list_files:
                    fileid = open(filename, 'a')
                else: # create the file
                    fileid = open(filename, 'w')
                    list_files.append(filename)
                #image_key = os.path.splitext(anno.image_name)[0]
                image_key = anno.image_name
                fileid.write('{0}\t{1}\n'.format(image_key, 1)) # positive
                fileid.close()
                if positive_set.has_key(label):
                    positive_set[label].append(image_key)
                else:
                    positive_set[label] = [image_key]
        db_input.close()
    # generate the categories.txt file
    filename = params.output_dir + '/categories.txt'
    logging.info('***** Creating file {0}'.format(filename))
    fileid = open(filename, 'w')
    for label in positive_set.keys():
        fileid.write('{0}\n'.format(label))
    fileid.close()
    return positive_set

def run_generate_neg(params, positive_set):
    num_neg_from_class_j = \
            int(params.num_neg_per_class/float(len(positive_set)-1))
    negative_set = {}
    for label_i in positive_set.keys():
        filename = params.output_dir + '/' + \
                                label_i + '_' + params.append_name + '.txt'
        logging.info('***** Processing file {0}'.format(filename))
        # open file in append mode (assuming pos example already run)
        fileid = open(filename, 'a')
        for label_j in positive_set.keys():
            # images containing other objects will be negative for label_i
            if label_j != label_i:
                # select num_neg_from_class_j random images
                idxs = range(len(positive_set[label_j]))
                if params.append_name == 'train':
                    random.shuffle(idxs)
                    idxs = idxs[:num_neg_from_class_j]
                negative_set[label_i] = [positive_set[label_j][i] for i in idxs]
                for i in idxs:
                    image_key = positive_set[label_j][i]
                    logging.info('(-) Negative Image {0}'.\
                                        format(image_key))
                    # add to neg list
                    if negative_set.has_key(label_i):
                        negative_set[label_i].append(image_key)
                    else:
                        negative_set[label_i] = [image_key]
                    # add to file
                    fileid.write('{0}\t{1}\n'.format(image_key, -1)) # negative
        fileid.close()
    return negative_set


if __name__ == "__main__":
    # set seed
    random.seed(0)
    # load configurations and parameters
    conf = Configuration()
    params = Params()
    # experiment name
    params.exp_name = 'ILSVRC2013_clsloc'
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp03_07' # exp03_07 or exp03_06
    params.append_name = 'train' # 'train' or 'val'
    params.subset = True
    params.num_neg_per_class = 5000 # val and test set, we keep them all
    if params.subset: # subset
        params.exp_name = params.exp_name + '/200RND'
    else:
        params.exp_name = params.exp_name + '/1000'
    # default Configuration, image and label files
    params.conf = conf
    # method for calculating the confidence
    params.heatextractor_confidence_tech = 'full_obf_positive'
    # input/output directory
    params.output_dir = './' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # parallelize the script on Anthill?
    params.run_on_anthill = True
    # list of tasks to execute
    params.task = []
    logging.info('Started')
    # RUN THE EXPERIMENT
    positive_set = run_generate_pos(params)
    negative_set = run_generate_neg(params, positive_set)
