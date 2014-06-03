import cPickle as pickle
import bsddb
import logging
import numpy as np
import matplotlib.pyplot as plt

from util import *
from network import *
from configuration import *
from bbox import *
from annotatedimage import *

class Params:
    def __init__(self):
        pass

if __name__=='__main__':
    # load configurations and parameters
    conf = Configuration()
    params = Params()
    # input (GT AnnotatatedImages)
    params.exp_name_input = 'exp03_07'
    params.shard = 1
    # Choose net
    params.classifier = 'CAFFE'
    params.center_only = True
    # list of layers to be visualized
    params.layers = ['data', 'conv1', 'pool1', 'norm1', \
                     'conv2', 'pool2', 'norm2', \
                     'conv3', 'conv4', 'conv5', 'pool5',\
                     'fc6', 'fc7', 'fc8', 'prob']
    # Num elements in batch (for decaf/caffe eval)
    params.batch_sz = 1
    # default Configuration, image and label files
    params.conf = conf
    # input/output directory
    #params.output_dir = conf.experiments_output_directory \
    #                    + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input
    # Load network
    assert params.classifier=='CAFFE', 'Visualization works only with CAFFE'
    netParams = NetworkCaffeParams(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only)
    net = Network.create_network(netParams)
    # load an shard
    inputdb = params.input_dir + '/%05d'%params.shard + '.db'
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    # get the image key
    image_key = db_input.keys()[4]
    # get database entry
    anno = pickle.loads(db_input[image_key])
    # get stuff from database entry
    img = anno.get_image()
    # extract features
    net_features = net.extract_all(img)
    # visualize features
    fig1 = plt.figure(0)
    net.visualize_features(net_features, params.layers, fig1)
    fig1.show()
    # Gray-out a specific region
    image_obf = img.copy()
    xmin, ymin, xmax, ymax = [152, 148, 305, 300]
    image_obf[ymin:ymax, xmin:xmax, 0] = net.get_mean_img()[0]
    image_obf[ymin:ymax, xmin:xmax, 1] = net.get_mean_img()[1]
    image_obf[ymin:ymax, xmin:xmax, 2] = net.get_mean_img()[2]
    # extract features
    net_features_obf = net.extract_all(image_obf)
    # visualize features
    fig2 = plt.figure(1)
    net.visualize_features(net_features_obf, params.layers, fig2)
    fig2.show()
    # close the db
    db_input.close()
