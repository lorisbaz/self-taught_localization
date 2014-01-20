"""
Configuration class to store paths (to ImageNet, Decaf, etc....),
environment information, etc...
"""

import os

class Configuration:
    """
    The following public vaiables (constants) are available:
    - ilsvrc2012_train_images
           directory storing the training images
    - ilsvrc2012_val_images
           directory storing the validation images
    - ilsvrc2012_classid_wnid_words
           containing the classid, wnid and words,
           according the official ILSVRC2012 specification

    - ilsvrc2012_decaf_model_spec
           the meta file for decaf
    - ilsvrc2012_decaf_model
           the trained model for decaf

    - ilsvrc2012_caffe_model_spec
    - ilsvrc2012_caffe_model
    - ilsvrc2012_caffe_wnids_words
            the ordered list of wnids and words used by the caffe

    - pascal2007_images
           directory storing the images for PASCAL VOC 2007
    """

    def __init__(self):
        if os.uname()[1] == 'anthill.cs.dartmouth.edu':
            ilsvrc2012_root = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012'
            self.ilsvrc2012_train_images = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val'
            self.ilsvrc2012_classid_wnid_words = \
                ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
                ilsvrc2012_root + '/decaf_model_131205/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
                ilsvrc2012_root +'/decaf_model_131205/imagenet.decafnet.epoch90'
            self.ilsvrc2012_caffe_model_spec = \
                '/home/anthill/vlg/caffe_131211/caffe/examples'\
                '/imagenet_deploy.prototxt'
            self.ilsvrc2012_caffe_model = \
                ilsvrc2012_root + '/caffe_model_131211'\
                '/caffe_reference_imagenet_model'
            self.ilsvrc2012_caffe_wnids_words = \
                '/home/anthill/vlg/caffe_131211/caffe/examples/synset_words.txt'
            self.pascal2007_images = \
                '/home/data0/vlg/Data/Images' \
                '/PASCAL_VOC_2007/VOCdevkit/VOC2007/JPEGImages'
        elif os.uname()[1] == 'alessandro-Linux':
            ilsvrc2012_root = '/home/alessandro/Data/ILSVRC2012'
            self.ilsvrc2012_train_images = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val'
            self.ilsvrc2012_classid_wnid_words = \
                ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
                '/home/alessandro/Data/decaf_ImageNet_model'\
                '/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
                '/home/alessandro/Data/decaf_ImageNet_model'\
                '/imagenet.decafnet.epoch90'
            self.ilsvrc2012_caffe_model_spec = \
                '/home/alessandro/Code/caffe_131211/caffe/examples'\
                '/imagenet_deploy.prototxt'
            self.ilsvrc2012_caffe_model = \
                '/home/alessandro/Data/ILSVRC2012/caffe_model'\
                '/caffe_reference_imagenet_model'
            self.ilsvrc2012_caffe_wnids_words = \
                '/home/alessandro/Code/caffe_131211/caffe/examples'\
                '/synset_words.txt'
            self.pascal2007_images = '/home/alessandro/Data/VOCdevkit/VOC2007'\
            '/JPEGImages'
        elif os.uname()[1] == 'loris-linux':
	    ilsvrc2012_root = '/home/lbazzani/DATASETS/ILSVRC2012'
            self.ilsvrc2012_train_images = ilsvrc2012_root + '/img_train'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/img_val'
            self.ilsvrc2012_classid_wnid_words = \
            ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
            '/home/lbazzani/CODE/DATA/decaf_ImageNet_model/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
            '/home/lbazzani/CODE/DATA/decaf_ImageNet_model/imagenet.decafnet.epoch90'
            self.ilsvrc2012_caffe_model_spec = 'TODO'
            self.ilsvrc2012_caffe_model = 'TODO'
            self.ilsvrc2012_caffe_wnids_words = 'TODO'
            self.pascal2007_images = '/home/lbazzani/DATASETS/VOC2007'\
            '/JPEGImages'
        else:
            raise ValueError('The current machine is not supported')


