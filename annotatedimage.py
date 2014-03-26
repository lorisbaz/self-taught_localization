import sys

import util

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
    - stats: dictionary {'name'} -> Stats
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

