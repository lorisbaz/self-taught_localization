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
            ilsvrc2012_root + '/decaf_model_131205/imagenet.decafnet.epoch90'
            self.pascal2007_images = '/home/data0/vlg/Data/Images'\
            '/PASCAL_VOC_2007/VOCdevkit/VOC2007/JPEGImages'
        elif os.uname()[1] == 'alessandro-Linux':
            ilsvrc2012_root = '/home/alessandro/Data/ILSVRC2012'
            self.ilsvrc2012_train_images = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val'
            self.ilsvrc2012_classid_wnid_words = \
            ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
            '/home/alessandro/Data/decaf_ImageNet_model/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
            '/home/alessandro/Data/decaf_ImageNet_model/imagenet.decafnet.epoch90'
            self.pascal2007_images = '/home/alessandro/Data/VOCdevkit/VOC2007'\
            '/JPEGImages'
        elif os.uname()[1] == 'LORIS_COMPUTER':
            # TODO
            raise NotImplementedError()
        else:
            raise ValueError('The current machine is not supported')


