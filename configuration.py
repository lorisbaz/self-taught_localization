"""
Configuration class to store paths (to ImageNet, Decaf, etc....),
environment information, etc...
"""

import os

class Configuration:
    """
    The following public vaiables (constants) are available:
    - ilsvrc2012_root_images_dir
           root directory storing the all the images
    - ilsvrc2012_train_images_dir
           directory storing the training images
    - ilsvrc2012_val_images_dir
           directory storing the validation images
    - ilsvrc2012_test_images_dir
           directory storing the test images
    
    - ilsvrc2012_val_images
           list of validation images (each entry is "val/imagename.JPEG")
    - ilsvrc2012_val_labels
           list of labels for the validation images (from 1 to 1000)
           according the official ILSVRC2012 specification

    - ilsvrc2012_classid_wnid_words
           containing the classid, wnid and words (separated by a tab),
           according the official ILSVRC2012 specification
    - ilsvrc2012_train_box_gt
           directory containing the XML files of the training bbox information
    - ilsvrc2012_val_box_gt
           directory containing the XML files of the validation bbox information

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

    - experiments_output_directory
           root directory storing all the partial and final results
    - ilsvrc2012_segm_results_dir
           directory storing the segmentation results
    """

    def __init__(self):
        if os.uname()[1] == 'anthill.cs.dartmouth.edu':
            ilsvrc2012_root = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012'
            self.ilsvrc2012_root_images_dir = ilsvrc2012_root
            self.ilsvrc2012_train_images_dir = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images_dir = ilsvrc2012_root + '/val'
            self.ilsvrc2012_test_images_dir = ilsvrc2012_root + '/test'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val_images.txt'
            self.ilsvrc2012_val_labels = ilsvrc2012_root + '/val_labels.txt'
            self.ilsvrc2012_train_box_gt = ilsvrc2012_root + '/bbox_train'
            self.ilsvrc2012_val_box_gt = ilsvrc2012_root + '/bbox_val'
            self.ilsvrc2012_classid_wnid_words = \
                ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
                ilsvrc2012_root + '/decaf_model_131205/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
                ilsvrc2012_root +'/decaf_model_131205/imagenet.decafnet.epoch90'
            self.ilsvrc2012_caffe_model_spec = \
                '/home/anthill/vlg/caffe/examples'\
                '/imagenet_deploy_GRAYOBFUSCATION.prototxt'
            self.ilsvrc2012_caffe_model = \
                ilsvrc2012_root + '/caffe_model_131211'\
                '/caffe_reference_imagenet_model'
            self.ilsvrc2012_caffe_avg_image = \
                '/home/anthill/vlg/caffe/python/caffe/imagenet'\
                '/ilsvrc_2012_mean.npy'
            self.ilsvrc2012_caffe_wnids_words = \
                '/home/anthill/vlg/caffe_131211/caffe/examples/synset_words.txt'
            self.pascal2007_images = \
                '/home/data0/vlg/Data/Images' \
                '/PASCAL_VOC_2007/VOCdevkit/VOC2007/JPEGImages'
            self.experiments_output_directory = \
                '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation'
       	    self.ilsvrc2012_segm_results_dir = \
	       self.experiments_output_directory + '/segment_ILSVRC2012' 
	elif os.uname()[1] == 'alessandro-Linux':
            ilsvrc2012_root = '/home/alessandro/Data/ILSVRC2012'
            self.ilsvrc2012_root_images_dir = ilsvrc2012_root
            self.ilsvrc2012_train_images_dir = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images_dir = ilsvrc2012_root + '/val'
            self.ilsvrc2012_test_images_dir = ilsvrc2012_root + '/test'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val_images.txt'
            self.ilsvrc2012_val_labels = ilsvrc2012_root + '/val_labels.txt'
            self.ilsvrc2012_train_box_gt = ilsvrc2012_root + '/bbox_train'
            self.ilsvrc2012_val_box_gt = ilsvrc2012_root + '/bbox_val'
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
            self.ilsvrc2012_caffe_avg_image = \
                '/home/alessandro/Code/caffe_131211/caffe/python/caffe/imagenet'\
                '/ilsvrc_2012_mean.npy'
            self.ilsvrc2012_caffe_wnids_words = \
                '/home/alessandro/Code/caffe_131211/caffe/examples'\
                '/synset_words.txt'
            self.pascal2007_images = '/home/alessandro/Data/VOCdevkit/VOC2007'\
                '/JPEGImages'
            self.experiments_output_directory = \
                '/home/alessandro/Data_projects/grayobfuscation'
       	    self.ilsvrc2012_segm_results_dir = \
	       self.experiments_output_directory + '/segment_ILSVRC2012' 
	elif os.uname()[1] == 'lbazzani-desk':
            ilsvrc2012_root = '/home/lbazzani/DATASETS/ILSVRC2012'
            self.ilsvrc2012_root_images_dir = ilsvrc2012_root
            self.ilsvrc2012_train_images_dir = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images_dir = ilsvrc2012_root + '/val'
            self.ilsvrc2012_test_images_dir = ilsvrc2012_root + '/test'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val_images.txt'
            self.ilsvrc2012_val_labels = ilsvrc2012_root + '/val_labels.txt'
            self.ilsvrc2012_train_box_gt = ilsvrc2012_root + '/bbox_train'
            self.ilsvrc2012_val_box_gt = ilsvrc2012_root + '/bbox_val'
            self.ilsvrc2012_classid_wnid_words = \
                ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
               '/home/lbazzani/CODE/DATA/decaf_ImageNet_model'\
               '/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
               '/home/lbazzani/CODE/DATA/decaf_ImageNet_model'\
               '/imagenet.decafnet.epoch90'
            self.ilsvrc2012_caffe_model_spec = \
                '/home/lbazzani/CODE/DATA/caffe_ImageNet_model'\
                '/imagenet_deploy.prototxt'
            self.ilsvrc2012_caffe_model = \
                '/home/lbazzani/CODE/DATA/caffe_ImageNet_model'\
                '/caffe_reference_imagenet_model'
            self.ilsvrc2012_caffe_avg_image = \
                '/home/lbazzani/CODE/DATA/caffe_ImageNet_model'\
                '/ilsvrc_2012_mean.npy'
            self.ilsvrc2012_caffe_wnids_words = \
                '/home/lbazzani/CODE/DATA/caffe_ImageNet_model'\
                '/synset_words.txt'
            self.pascal2007_images = '/home/lbazzani/DATASETS/VOC2007'\
               '/JPEGImages'
            self.experiments_output_directory = \
               '/home/lbazzani/CODE/DATA/obfuscation_results'
       	    self.ilsvrc2012_segm_results_dir = \
	       self.experiments_output_directory + '/segment_ILSVRC2012' 
	else:
            raise ValueError('The current machine is not supported')


