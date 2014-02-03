import numpy as np
import sys

class BBox:
    """
    Class that represent a bbox, with some utility functions.
    The following public fields are available:
    - xmin, ymin, xmax, ymax, confidence (all float)
    """

    def __init__(self, xmin, ymin, xmax, ymax, confidence):
        assert xmin >= 0.0
        assert ymin >= 0.0
        assert xmax <= 1.0
        assert ymax <= 1.0
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.confidence = confidence

    def area(self):
        return np.abs(self.xmax-self.xmin)*np.abs(self.ymax-self.ymin)
