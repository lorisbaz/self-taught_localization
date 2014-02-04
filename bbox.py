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

    def __init__(self, xmin, ymin, xmax, ymax, confidence):
        assert xmin >= 0
        assert ymin >= 0
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence

    def __str__(self):
        return 'bbox: [{0}. {1}. {2}. {3}] conf: {4}'.format( \
                  self.xmin, self.ymin, self.xmax, self.ymax, self.confidence)

    def area(self):
        return np.abs(self.xmax-self.xmin)*np.abs(self.ymax-self.ymin)

    def normalize_to_outer_box(self, outer_box):
        """
        Normalize the rectangle defining the bbox to have 
        0.0 <= width/height/area <= 1.0.
        outer_box is (height, width)
        """
        self.xmin /= float(outer_box[1])
        self.ymin /= float(outer_box[0])
        self.xmax /= float(outer_box[1])
        self.ymax /= float(outer_box[0])

