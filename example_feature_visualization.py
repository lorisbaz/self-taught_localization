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
    params.which_image = 4
    params.topC = 5
    # Choose net
    params.classifier = 'CAFFE'
    params.center_only = True
    # list of layers to be visualized
#    params.layers = ['data', 'conv1', 'pool1', 'norm1', \
#                     'conv2', 'pool2', 'norm2', \
#                     'conv3', 'conv4', 'conv5', 'pool5',\
#                     'fc6', 'fc7', 'fc8', 'prob']
    params.layers = ['data', 'conv1',  'norm1', \
                     'conv2', 'norm2', \
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
    label_list = net.get_labels()
    # load an shard
    inputdb = params.input_dir + '/%05d'%params.shard + '.db'
    logging.info('Opening ' + inputdb)
    db_input = bsddb.btopen(inputdb, 'r')
    # get the image key
    image_key = db_input.keys()[4]
    # get database entry and GT label
    anno = pickle.loads(db_input[image_key])
    GT_ID = net.get_label_id(anno.gt_objects.keys()[0])
    GT_name = net.get_label_desc(anno.gt_objects.keys()[0])
    # get stuff from database entry
    img = anno.get_image()
    # extract features
    logging.info('Original image')
    net_features = net.extract_all(img)
    sort_guess = np.argsort(net_features['prob'].data[0], axis=0)[::-1]
    for i in range(params.topC):
        class_name = net.get_label_desc(label_list[sort_guess[i]])
        logging.info('ID prediction: {0} and score {1}'.\
                format(class_name, \
                net_features['prob'].data[0][sort_guess[i]]))
    logging.info('Score using the GT label {0}'.format(\
                net_features['prob'].data[0][GT_ID]))
    # visualize features
    fig1 = plt.figure(0)
    net.visualize_features(net_features, params.layers, fig1, \
                            dump_image_path = './main_figure_paper/original')
    fig1.show()
    # Gray-out a specific region (containing the obj)
    image_obf = img.copy()
    xmin, ymin, xmax, ymax = [152, 148, 305, 300]
    image_obf[ymin:ymax, xmin:xmax, 0] = net.get_mean_img()[0]
    image_obf[ymin:ymax, xmin:xmax, 1] = net.get_mean_img()[1]
    image_obf[ymin:ymax, xmin:xmax, 2] = net.get_mean_img()[2]
    # extract features
    logging.info('Obfuscating the main object')
    net_features_obf = net.extract_all(image_obf)
    sort_guess = np.argsort(net_features_obf['prob'].data[0], axis=0)[::-1]
    for i in range(params.topC):
        class_name = net.get_label_desc(label_list[sort_guess[i]])
        logging.info('ID prediction: {0} and score {1}'.\
                format(class_name, \
                net_features_obf['prob'].data[0][sort_guess[i]]))
    logging.info('Score using the GT label {0}'.format(\
                net_features_obf['prob'].data[0][GT_ID]))
    # visualize features
    fig2 = plt.figure(1)
    net.visualize_features(net_features_obf, params.layers, fig2, \
                            dump_image_path = './main_figure_paper/obf_obj')
    fig2.show()
    # Gray-out a specific region (not containing obj)
    image_obf = img.copy()
    xmin, ymin, xmax, ymax = [0, 0, 250, 150]
    image_obf[ymin:ymax, xmin:xmax, 0] = net.get_mean_img()[0]
    image_obf[ymin:ymax, xmin:xmax, 1] = net.get_mean_img()[1]
    image_obf[ymin:ymax, xmin:xmax, 2] = net.get_mean_img()[2]
    # extract features
    logging.info('Obfuscating other parts of the image')
    net_features_obf = net.extract_all(image_obf)
    sort_guess = np.argsort(net_features_obf['prob'].data[0], axis=0)[::-1]
    for i in range(params.topC):
        class_name = net.get_label_desc(label_list[sort_guess[i]])
        logging.info('ID prediction: {0} and score {1}'.\
                format(class_name, \
                net_features_obf['prob'].data[0][sort_guess[i]]))
    logging.info('Score using the GT label {0}'.format(\
                net_features_obf['prob'].data[0][GT_ID]))
    # visualize features
    fig3 = plt.figure(2)
    net.visualize_features(net_features_obf, params.layers, fig3, \
                            dump_image_path = './main_figure_paper/obf_other')
    fig3.show()
    # close the db
    db_input.close()
