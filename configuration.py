"""
Configuration class to store paths (to ImageNet, Decaf, etc....),
environment information, etc...
"""

import os

class Configuration:
    """
    The following public vaiables (constants) are available:

    ***** ILSVRC 2012 *****
    - ilsvrc2012_root_images_dir
           root directory storing all the images
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

    - ilsvrc2012_segm_results_dir
           directory storing the segmentation results

    ***** ILSVRC 2013 detection *****
    - ilsvrc2013_root_images_dir
           root directory storing all the images
    - ilsvrc2013_train_images_dir
           directory storing the training images
    - ilsvrc2013_val_images_dir
           directory storing the validation images
    - ilsvrc2013_test_images_dir
           directory storing the test images

    - ilsvrc2013_val_images
           list of validation images (each entry is "val/imagename.JPEG")

    - ilsvrc2013_classid_wnid_words
           containing the classid, wnid and words (separated by a tab),
           according the official ILSVRC2012 specification
    - ilsvrc2013_train_box_gt
           directory containing the XML files of the training bbox information
    - ilsvrc2013_val_box_gt
           directory containing the XML files of the validation bbox information

    - ilsvrc2013_decaf_model_spec
           the meta file for decaf
    - ilsvrc2013_decaf_model
           the trained model for decaf

    - ilsvrc2013_caffe_model_spec
    - ilsvrc2013_caffe_model
    - ilsvrc2013_caffe_wnids_words
            the ordered list of wnids and words used by the caffe

    ***** PASCAL VOC 2007 *****
    - pascal2007_root_dir
           directory containing the VOC devkit (which includes images, XMLs,etc)
    - pascal2007_images_dir
           directory storing the images (extension: ".jpg")
    - pascal2007_image_file_extension
           string: ".jpg"
    - pascal2007_classes
           list of strings, representing the 20 classes of VOC
    - pascal2007_sets_dir
           directory containing the sets.
           ** The following textfiles-sets are available:
           train.txt, val.txt, test.txt, trainval.txt
           where each line an image-key (name of the image, without .jpg).
           ** Also, each class has, for the classification task:
           classname_train.txt, classname_val.txt, classname_test.txt,
           classname_trainval.txt, each line being "<image-key> <1,-1,0>"
           (1 for positive, -1 for negative, 0 for difficult)
    - pascal2007_annotations_dir
           directory containing the XML files

    ***** GRAYOBFUSCATION PROJECT *****
    - experiments_output_directory
           root directory storing all the partial and final results
    """

    def __init__(self, load_Girshick_caffe_model = False):
        hostname = os.uname()[1]
        if  'anthill' in hostname or \
            'bokken' in hostname or \
            'ennead' in hostname or \
            'gridiron' in hostname or \
            'katana' in hostname or \
            'killington' in hostname or \
            'nodachi' in hostname or \
            'tanto' in hostname:
            # ***** ILSVRC 2012 Localization *****
            ilsvrc2012_root = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012'
            self.ilsvrc2012_root_images_dir = ilsvrc2012_root
            self.ilsvrc2012_train_images_dir = ilsvrc2012_root + '/train'
            self.ilsvrc2012_val_images_dir = ilsvrc2012_root + '/val'
            self.ilsvrc2012_test_images_dir = ilsvrc2012_root + '/test'
            self.ilsvrc2012_train_images = ilsvrc2012_root + '/train_images.txt'
            self.ilsvrc2012_val_images = ilsvrc2012_root + '/val_images.txt'
            self.ilsvrc2012_train_labels = ilsvrc2012_root + '/train_labels.txt'
            self.ilsvrc2012_val_labels = ilsvrc2012_root + '/val_labels.txt'
            self.ilsvrc2012_train_box_gt = ilsvrc2012_root + '/bbox_train'
            self.ilsvrc2012_val_box_gt = ilsvrc2012_root + '/bbox_val'
            self.ilsvrc2012_classid_wnid_words = \
                ilsvrc2012_root + '/classid_wnid_words.txt'
            self.ilsvrc2012_decaf_model_spec = \
                ilsvrc2012_root + '/decaf_model_131205/imagenet.decafnet.meta'
            self.ilsvrc2012_decaf_model = \
                ilsvrc2012_root +'/decaf_model_131205/imagenet.decafnet.epoch90'
            if not load_Girshick_caffe_model:
                self.ilsvrc2012_caffe_model_spec = \
                    './imagenet_deploy_GRAYOBFUSCATION.prototxt'
                self.ilsvrc2012_caffe_model = \
                    ilsvrc2012_root + '/caffe_model_131211'\
                    '/caffe_reference_imagenet_model'
            else:
                self.ilsvrc2012_caffe_model_spec = \
                    './rcnn_batch_256_output_fc7_GRAYOBFUSCATION.prototxt'
                self.ilsvrc2012_caffe_model = \
                    '/home/anthill/vlg/rcnn/data/caffe_nets' \
                    '/ilsvrc_2012_train_iter_310k'
            self.ilsvrc2012_caffe_avg_image = \
                ilsvrc2012_root + '/caffe_model_131211'\
                '/ilsvrc_2012_mean.npy'
            self.ilsvrc2012_caffe_wnids_words = \
                ilsvrc2012_root + '/caffe_model_131211'\
                '/synset_words.txt'
       	    self.ilsvrc2012_segm_results_dir = \
                 '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation'\
                 '/segment_ILSVRC2012'
            # ***** ILSVRC 2013 Detection *****
            ilsvrc2013_root = \
                        '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2013/DET'
            ilsvrc2013_splits_root = './ILSVRC2013_det'
            self.ilsvrc2013_root_images_dir = ilsvrc2013_root
            self.ilsvrc2013_train_images_dir = ilsvrc2013_root + '/train'
            self.ilsvrc2013_val_images_dir = ilsvrc2013_root + '/val'
            self.ilsvrc2013_test_images_dir = ilsvrc2013_root + '/test'
            self.ilsvrc2013_train_images = \
                            ilsvrc2013_splits_root + '/train_{0}.txt'
            self.ilsvrc2013_val_images = \
                            ilsvrc2013_splits_root + '/val_{0}.txt'
            self.ilsvrc2013_test_images = \
                            ilsvrc2013_splits_root + '/test.txt'
            self.ilsvrc2013_train_box_gt = ilsvrc2013_root + '/bbox_train'
            self.ilsvrc2013_val_box_gt = ilsvrc2013_root + '/bbox_val'
            # We select the subset of labels that overlaps with ILSVRC2012,
            # decomment the second line if you want all the label
            self.ilsvrc2013_classid_wnid_words_overlap = \
                ilsvrc2013_root + '/label_subset_overlap_with_ILSVRC2012.txt'
            self.ilsvrc2013_classid_wnid_words = \
                ilsvrc2013_root + '/classid_wnid_words.txt'
            self.ilsvrc2013_decaf_model_spec = \
                ilsvrc2013_root + '/decaf_model_131205/imagenet.decafnet.meta'
            self.ilsvrc2013_decaf_model = \
                ilsvrc2013_root +'/decaf_model_131205/imagenet.decafnet.epoch90'
            if not load_Girshick_caffe_model:
                self.ilsvrc2013_caffe_model_spec = \
                    './imagenet_deploy_GRAYOBFUSCATION.prototxt'
                self.ilsvrc2013_caffe_model = \
                    ilsvrc2013_root + '/caffe_model_131211'\
                    '/caffe_reference_imagenet_model'
            else:
                self.ilsvrc2013_caffe_model_spec = \
                    './rcnn_batch_256_output_fc7_GRAYOBFUSCATION.prototxt'
                self.ilsvrc2013_caffe_model = \
                    '/home/anthill/vlg/rcnn/data/caffe_nets' \
                    '/ilsvrc_2012_train_iter_310k'
            self.ilsvrc2013_caffe_avg_image = \
                ilsvrc2013_root + '/caffe_model_131211'\
                '/ilsvrc_2012_mean.npy'
            self.ilsvrc2013_caffe_wnids_words = \
                ilsvrc2013_root + '/caffe_model_131211'\
                '/synset_words.txt'
       	    #self.ilsvrc2013_segm_results_dir = \
            #     '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation'\
            #     '/segment_ILSVRC2013'
            # ***** PASCAL VOC 2007 *****
            self.pascal2007_root_dir = \
                '/home/ironfs/scratch/vlg/Data/Images/PASCAL_VOC_2007'
            self.pascal2007_images_dir = \
                self.pascal2007_root_dir + '/JPEGImages'
            self.pascal2007_image_file_extension = '.jpg'
            self.pascal2007_classes = \
                 ['aeroplane','bicycle','bird','boat',\
                 'bottle','bus','car','cat',\
                 'chair','cow','diningtable','dog',\
                 'horse','motorbike','person','pottedplant',\
                 'sheep','sofa','train','tvmonitor']
            self.pascal2007_sets_dir = \
                self.pascal2007_root_dir + '/ImageSets/Main'
            self.pascal2007_annotations_dir = \
                self.pascal2007_root_dir + '/Annotations'
            # ***** GRAYOBFUSCATION PROJECT *****
            self.experiments_output_directory = \
                '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation'
	elif hostname == 'alessandro-Linux':
            # ***** ILSVRC 2012 *****
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
       	    self.ilsvrc2012_segm_results_dir = \
                '/home/alessandro/Data_projects/grayobfuscation/'\
                'segment_ILSVRC2012'
            # ***** ILSVRC 2013 Detection *****
            # TODOOO: now only anthill is supported
            # ***** PASCAL VOC 2007 *****
            self.pascal2007_root_dir = \
                '/home/alessandro/Data/VOCdevkit/VOC2007'
            self.pascal2007_images_dir = \
                self.pascal2007_root_dir + '/JPEGImages'
            self.pascal2007_image_file_extension = '.jpg'
            self.pascal2007_classes = \
                 ['aeroplane','bicycle','bird','boat',\
                 'bottle','bus','car','cat',\
                 'chair','cow','diningtable','dog',\
                 'horse','motorbike','person','pottedplant',\
                 'sheep','sofa','train','tvmonitor']
            self.pascal2007_sets_dir = \
                self.pascal2007_root_dir + '/ImageSets/Main'
            self.pascal2007_annotations_dir = \
                self.pascal2007_root_dir + '/Annotations'
            # ***** GRAYOBFUSCATION PROJECT *****
            self.experiments_output_directory = \
                '/home/alessandro/Data_projects/grayobfuscation'
	elif hostname == 'lbazzani-desk':
            # ***** ILSVRC 2012 *****
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
       	    self.ilsvrc2012_segm_results_dir = \
               '/home/lbazzani/CODE/DATA/obfuscation_results/'\
               'segment_ILSVRC2012'
            # ***** ILSVRC 2013 Detection *****
            # TODOOO: now only anthill is supported
            # ***** PASCAL VOC 2007 *****
            self.pascal2007_root_dir = \
                '/home/lbazzani/DATASETS/VOC2007'
            self.pascal2007_images_dir = \
                self.pascal2007_root_dir + '/JPEGImages'
            self.pascal2007_image_file_extension = '.jpg'
            self.pascal2007_classes = \
                 ['aeroplane','bicycle','bird','boat',\
                 'bottle','bus','car','cat',\
                 'chair','cow','diningtable','dog',\
                 'horse','motorbike','person','pottedplant',\
                 'sheep','sofa','train','tvmonitor']
            self.pascal2007_sets_dir = \
                self.pascal2007_root_dir + '/ImageSets/Main'
            self.pascal2007_annotations_dir = \
                self.pascal2007_root_dir + '/Annotations'
            # ***** GRAYOBFUSCATION PROJECT *****
            self.experiments_output_directory = \
               '/home/lbazzani/CODE/DATA/obfuscation_results'
	else:
            raise ValueError('The current machine is not supported')


