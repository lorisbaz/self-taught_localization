import copy
import numpy as np
import sys

import util
from featextractor import *
from bbox import *

class AnnotatedHeatmap:
    """
    Object defining an heatmap with annotations.
    The public fields are:
    - heatmap: nd-array.
    - description: string. A textual description of what this heatmap refers to.
    - type: string. the classname that generated this heatmap.
    - spec: implementation-specific data, containing more information regarding
            how the heatmap has been extracted.
    """
    def __init__(self):
        self.heatmap = None
        self.description = ''
        self.type = ''
        self.specs = None

class AnnotatedObject:
    """
    Everything regarding the annotation of an object.
    The public fields are:
    - label: string (i.e. 'n0001000')
    - confidence: float.  The confidence value associated with this object
           Normally it refers to the full-image confidence. If the confidence
           is not available, this field is None.
    - bboxes: list of BBox objects (if any)
    - heatmaps: list of AnnotatedHeatmap objects
    """
    def __init__(self, label='', confidence=None):
        self.label = label
        self.confidence = confidence
        self.bboxes = []
        self.heatmaps = []

    def __str__(self):
        out = 'label: ' + str(self.label) + '; '
        for bbox in self.bboxes:
            out += str(bbox) + ' '
        return out

class AnnotatedImage:
    """
    All the information regarding an image.
    The public fields are:
    - image_jpeg: array of bytes.
                  An array of bytes containing the image encoded in JPEG.
                  Use the methods set/get_image to set/get this field.
    - image_width, image_height: int. This MUST match the size of image_jpeg
    - image_name: string
                  The unique file identifier of the image
                  (i.e. 'val/ILSVRC2012_val_00000001.JPEG')
    - gt_objects: dictionary  {'label'} -> AnnotatedObject
                  Note that the confidence values must be set (with any value)
                  if you want to use the method get_gt_label()
    - pred_objects: dictionary {'name'} -> ({'label'} -> AnnotatedObject)
    - crop_description: string, containing a description regarding how the image
                        has been generated from its original version
    - segmentation_name: string, denoting the unique name of the segmentation
                         mask used for this image.
    - stats: dictionary {'name'} -> ({'label'} -> Stats)
    - features: {'feat_extractor_module'} -> data where data is
                   a private feature-dependent object.
    """
    def __init__(self):
        self.image_jpeg = ''
        self.image_width = 0
        self.image_height = 0
        self.image_name = ''
        self.gt_objects = {}
        self.pred_objects = {}
        self.crop_description = ''
        self.segmentation_name = ''
        self.stats = {}
        self.features = {}

    def __str__(self):
        out = '{0}:[{1} x {2}]\n'.format(self.image_name, \
                                         self.image_height, \
                                         self.image_width)
        out += 'gt_objects:\n'
        for label, obj in self.gt_objects.iteritems():
            out += '  ' + str(obj)
        return out

    def set_image(self, img):
        """
        Set the image, given a ndarray-image
        """
        self.image_jpeg = util.convert_image_to_jpeg_string(img)
        self.image_width = img.shape[1]
        self.image_height = img.shape[0]

    def get_image(self):
        """
        Return a ndarray-image
        """
        img = util.convert_jpeg_string_to_image(self.image_jpeg)
        assert self.image_width == img.shape[1]
        assert self.image_height == img.shape[0]
        return img

    def get_gt_label(self):
        """
        Return the top-scoring (full image) gt label.
        """
        label = ''
        max_conf = -sys.float_info.max
        for key, obj in self.gt_objects.iteritems():
            assert key == obj.label
            if (obj.confidence != None) and (obj.confidence > max_conf):
                label = obj.label
                max_conf = obj.confidence
        return label

    def set_stats(self):
        self.stats = {}

    def extend_pred_objects(self, anno, classifier):
        """
        This function extend the predicted objects in self with the pred_objs
        It keeps the label and confidence of self in case of collision.
        """
        for eachkey in anno.pred_objects[classifier].keys():
            # check existing key
            if self.pred_objects[classifier].has_key(eachkey): # collision
                self.pred_objects[classifier][eachkey].bboxes.extend( \
                          anno.pred_objects[classifier][eachkey].bboxes)
                self.pred_objects[classifier][eachkey].heatmaps.extend( \
                          anno.pred_objects[classifier][eachkey].heatmaps)
            else: # add the key
                self.pred_objects[classifier][eachkey] = \
                                    anno.pred_objects[classifier][eachkey]

    def export_pred_bboxes_to_text(self, name_pred_objects, \
                                   max_num_bboxes = sys.maxint):
        """
        Export the predicted bboxes to a text representation with multi lines,
        each line, with the following tab-separated fields:
        <image_name image_width image_height label full_image_confidence  ...
               xmin ymin xmax ymax bbox_confidence>

        If max_num_bboxes is set, for each image and class label
        we sort the bboxes by 
        confidence and we export only the top-max_num_bboxes bboxes per label.
        """
        assert name_pred_objects in self.pred_objects
        out = ''
        # for each AnnotatedObject ....
        for label in self.pred_objects[name_pred_objects]:
            anno_object = self.pred_objects[name_pred_objects][label]
            full_image_confidence = anno_object.confidence
            try: # if the conf is not a number, set it to zero
                full_image_confidence = float(full_image_confidence)
            except:
                full_image_confidence = -sys.float_info.max
            # for each bbox ...
            bboxes = copy.deepcopy(anno_object.bboxes)
            for bb in bboxes:
                try: # if the conf is not a number, set it to zero
                    bb.confidence = float(bb.confidence)
                except:
                    bb.confidence = -sys.float_info.max
            bboxes = sorted(bboxes, key = lambda bb: -bb.confidence)
            bboxes = bboxes[0:min(max_num_bboxes, len(bboxes))]
            for bbox in bboxes:
                bbox_confidence = bbox.confidence
                line = '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\t{6}\t{7}\t{8}\t{9}\n'\
                       .format( \
                       self.image_name, self.image_width, self.image_height, \
                       anno_object.label, full_image_confidence, \
                       bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax, \
                       bbox_confidence)
                out = out + line
        return out

    def extract_features(self, bboxes):
        """
        It extracts the feature vectors from the given list of bbboxes.
        It returns a np.ndarray matrix of size [num_bboxes, num_dims].
        Note that you must register a FeatureExtractor module first,
        using the register_feature_extractor() method.
        """
        # check input
        assert hasattr(self, 'feature_extractor_') and self.feature_extractor_, \
            'You must register a FeatureExtractor module'
        assert isinstance(bboxes, list)
        for bb in bboxes:
            assert isinstance(bb, BBox)
        # extract the features using the registered module
        feats = self.feature_extractor_.extract(bboxes)
        if self.save_features_cache_:
            self.features[self.feature_extractor_.name] = \
                     self.feature_extractor_.get_cache()
        # check the output and return
        assert isinstance(feats, np.ndarray)
        assert feats.shape[0] == len(bboxes)
        return feats

    def register_feature_extractor(self, feature_extractor_params, \
                                   save_features_cache = False):
        """
        Build and register a FeatureExtractor module, that will be
        used to extract the features from the image.
        The input must be a subclass of FeatureExtractorParams
        If save_features is True, the cache of the feature extractor
        module will be saved in the features field.
        """
        # check the input
        assert isinstance(feature_extractor_params, FeatureExtractorParams)
        if not hasattr(self, 'features'):
            self.features = {}
        if not hasattr(self.features, feature_extractor_params.name):
            self.features[feature_extractor_params.name] = {}
        if not hasattr(self, 'feature_extractor_'):
            self.feature_extractor_ = None
        assert not self.feature_extractor_, 'Already present a FeatExtractor'
        # register
        self.feature_extractor_ = FeatureExtractor.create_feature_extractor( \
                    self, feature_extractor_params)
        self.save_features_cache_ = save_features_cache


    
	

