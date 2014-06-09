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
    params.topC = 5
    # Choose net
    params.classifier = 'CAFFE'
    params.center_only = True
    # list of layers to be visualized
#    params.layers = ['data', 'conv1', 'pool1', 'norm1', \
#                     'conv2', 'pool2', 'norm2', \
#                     'conv3', 'conv4', 'conv5', 'pool5',\
#                     'fc6', 'fc7', 'fc8', 'prob']
    params.layers = ['data',  'norm1', 'norm2', \
                     'conv3', 'conv4', 'pool5',\
                     'fc8']
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
    for shard in range(25, 1000):
        logging.info('--- Processing shard {0}'.format(shard))
        # load an shard
        inputdb = params.input_dir + '/%05d'%shard + '.db'
        logging.info('Opening ' + inputdb)
        db_input = bsddb.btopen(inputdb, 'r')
        # get the image key
        image_keys = db_input.keys()
        for image_key in image_keys:
            # get database entry and a single GT object
            anno = pickle.loads(db_input[image_key])
            GT_label = anno.gt_objects.keys()[0] # <---- ONLY FIRST OBJ!!
            GT_ID = net.get_label_id(GT_label)
            GT_name = net.get_label_desc(GT_label)
            # get stuff from database entry
            img = anno.get_image()
            # extract features
            logging.info('Original image')
            net_features = net.extract_all(img)
            sort_guess = np.argsort(net_features['prob'].data[0],\
                                                            axis=0)[::-1]
            for i in range(params.topC):
                class_name = net.get_label_desc(label_list[sort_guess[i]])
                logging.info('ID prediction: {0} and score {1}'.\
                        format(class_name, \
                        net_features['prob'].data[0][sort_guess[i]]))
            score_orig = net_features['prob'].data[0][GT_ID].copy()
            logging.info('Score using the GT label {0}'.format(score_orig))
            # visualize features
            fig1 = plt.figure(0)
            fig1.clf()
            net.visualize_features(net_features, params.layers, fig1, \
                        subsample_kernels = 8, \
                        dump_image_path = './figure_add_mat/original' +\
                        image_key)
            fig1.show()
            # Gray-out a specific region (containing the obj)
            image_obf = img.copy()
            [height, width] = np.shape(image_obf)[0:2]
            # Get a single ground truth bbox
            bbox_this = anno.gt_objects[GT_label].bboxes[0] # <- ONLY FIRST BBOX!!
            bbox_this.rescale_to_outer_box(width, height)
            xmin, ymin, xmax, ymax = [bbox_this.xmin, bbox_this.ymin, \
                                        bbox_this.xmax, bbox_this.ymax]
            # Obfuscate
            if len(np.shape(image_obf))==3:
                image_obf[ymin:ymax, xmin:xmax, 0] = net.get_mean_img()[0]
                image_obf[ymin:ymax, xmin:xmax, 1] = net.get_mean_img()[1]
                image_obf[ymin:ymax, xmin:xmax, 2] = net.get_mean_img()[2]
            else:
                image_obf[ymin:ymax, xmin:xmax] = \
                                        np.mean(net.get_mean_img())
            # extract features
            logging.info('Obfuscating the main object')
            net_features_obf = net.extract_all(image_obf)
            sort_guess = np.argsort(net_features_obf['prob'].data[0],\
                                                            axis=0)[::-1]
            for i in range(params.topC):
                class_name = net.get_label_desc(label_list[sort_guess[i]])
                logging.info('ID prediction: {0} and score {1}'.\
                        format(class_name, \
                        net_features_obf['prob'].data[0][sort_guess[i]]))
            score_obf = net_features_obf['prob'].data[0][GT_ID]
            logging.info('Score using the GT label {0}'.format(score_obf))
            str_view = 'Drop = %.3f'%(score_orig-score_obf)
            # visualize features
            fig2 = plt.figure(1)
            fig2.clf()
            net.visualize_features(net_features_obf, params.layers, fig2, \
                        subsample_kernels = 8, \
                        dump_image_path = './figure_add_mat/obf_obj' +\
                        image_key, string_drop = str_view)
            fig2.show()
        db_input.close()
