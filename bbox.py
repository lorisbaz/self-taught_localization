import copy
import numpy as np
import sys

class BBox:
    """
    Class that represent a bbox, with some utility functions.
    The following public fields are available:
    - xmin, ymin, xmax, ymax
      These values define the rectangle defining the bbox,
      including xmin and ymin, while *excluding* xmax, ymax
      (so width = xmax-xmin)
    - confidence (float)
    """

    def __init__(self, xmin, ymin, xmax, ymax, confidence = 0.0):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence

    def __str__(self):
        xmin = float(self.xmin)
        ymin = float(self.ymin)
        xmax = float(self.xmax)
        ymax = float(self.ymax)
        confidence = float(self.confidence)
        return 'bbox: [{0:.2}. {1:.2}. {2:.2}. {3:.2}] conf: {4:.5} .'.format( \
                  xmin, ymin, xmax, ymax, confidence)

    def area(self):
        return np.abs(self.xmax-self.xmin)*np.abs(self.ymax-self.ymin)

    def normalize_to_outer_box(self, outer_box):
        """
        Normalize the rectangle defining the bbox to have 
        0.0 <= width/height/area <= 1.0, relative to the given BBox.
        Note: confidence is not modified.
        It returns self.
        """
        out_box_width = float(outer_box.xmax - outer_box.xmin)
        out_box_height = float(outer_box.ymax - outer_box.ymin)
        self.xmin = (self.xmin - outer_box.xmin) / out_box_width
        self.ymin = (self.ymin - outer_box.ymin) / out_box_height
        self.xmax = (self.xmax - outer_box.xmin) / out_box_width
        self.ymax = (self.ymax - outer_box.ymin) / out_box_height
        return self
    
    def intersect(self, bbox):
        """
        Intersection with the given bbox.
        Note: confidence is not modified.
        It returns self.
        """
        self.xmin = max(self.xmin, bbox.xmin)
        self.ymin = max(self.ymin, bbox.ymin)
        self.xmax = min(self.xmax, bbox.xmax)
        self.ymax = min(self.ymax, bbox.ymax)
        return self

    def jaccard_similarity(self, bbox):
        """
        Calculates the Jaccard similarity (the similarity used in the PASCAL VOC)
        """
        area_intersection = bbox.copy().intersect(self).area()
        area_union = self.area() + bbox.area() - area_intersection
        return area_intersection / float(area_union)

    def copy(self):
        return copy.deepcopy(self)


    

