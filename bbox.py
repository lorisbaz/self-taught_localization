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
        if isinstance(self.xmin, float):
            confidence = float(self.confidence)
            return 'bbox: [{0:.2} {1:.2} {2:.2} {3:.2}] conf: {4:.5} .'\
            .format(self.xmin, self.ymin, self.xmax, self.ymax, self.confidence)
        else:
            return 'bbox: [{0} {1} {2} {3}] conf: {4:.5} .'\
            .format(self.xmin, self.ymin, self.xmax, self.ymax, self.confidence)

    def area(self):
        return np.abs(self.xmax-self.xmin)*np.abs(self.ymax-self.ymin)

    def normalize_to_outer_box(self, outer_box):
        """
        Normalize the current integer rectangle defining the bbox to have
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

    def rescale_to_outer_box(self, width, height):
        """
        It converts the current 0-1 normalized Bbox, to
        absolute coordinates according the given rectangle.
        It returns self.
        """
        self.xmin *= float(width)
        self.ymin *= float(height)
        self.xmax *= float(width)
        self.ymax *= float(height)
        return self

    def convert_coordinates_to_integers(self):
        """
        It returns self.
        """
        self.xmin = int(self.xmin)
        self.ymin = int(self.ymin)
        self.xmax = int(self.xmax)
        self.ymax = int(self.ymax)
        return self

    def translate(self, x, y):
        """
        Translate the coordinates of the box, that will have
        (x, y) as the new origin.
        """
        self.xmin -= x
        self.ymin -= y
        self.xmax -= x
        self.ymax -= y
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
        if (self.xmin > self.xmax) or (self.ymin > self.ymax):
            self.xmin = 0.0
            self.ymin = 0.0
            self.xmax = 0.0
            self.ymax = 0.0
        return self

    def jaccard_similarity(self, bbox):
        """
        Calculates the Jaccard similarity (the similarity used in the
        PASCAL VOC)
        """
        area_intersection = bbox.copy().intersect(self).area()
        area_union = self.area() + bbox.area() - area_intersection
        return area_intersection / float(area_union)

    def copy(self):
        return copy.deepcopy(self)

    def get_coordinates_str(self):
        if isinstance(self.xmin, float):
            return '{0:.4}:{1:.4}:{2:.4}:{3:.4}'\
            .format(self.xmin, self.ymin, self.xmax, self.ymax)
        else:
            return '{0}:{1}:{2}:{3}'\
            .format(self.xmin, self.ymin, self.xmax, self.ymax)

    @staticmethod
    def non_maxima_suppression(bboxes, iou_threshold):
        """
        Run the classic NMS procedure: the input bboxes are sorted by their
        confidence scorse, while bboxes which have more than 'iou_threshold'
        overlap with a higher scoring bbox are consdered near-duplicates
        and removed.
        The method returns the remaining bboxes sorted
        by confidence.
        """
        assert iou_threshold >= 0.0
        if not bboxes:
            return []
        # make a copy of the bboxes, and sort them by confidence
        bboxes = copy.deepcopy(bboxes)
        bboxes.sort(key=lambda bb: bb.confidence, reverse=True)
        bboxes_out = []
        while len(bboxes) >= 1:
            bboxes_out.append(bboxes[0])
            bboxes = bboxes[1:]
            bboxes2 = []
            for bb in bboxes:
                if bb.jaccard_similarity(bboxes_out[-1]) <= iou_threshold:
                    bboxes2.append(bb)
            bboxes = bboxes2
        return bboxes_out
