import numpy as np
from scipy.misc import toimage

class Heatmap:
    """class for Heatmap"""

    def __init__(self, width, height):
        """
        width and height are two integers.
        """
        self.vals_ = np.zeros((height, width), np.float64)
        self.counts_ = np.zeros((height, width), np.int32)

    def add_val_rect(self, val, x, y, width = 1, height = 1, \
                     area_normalization = True):
        """
        Add val on the rectangle [x, y, x+width-1, y+width-1]
        """
        normalization_factor = 1.0
        if area_normalization:
            normalization_factor = width*height
        for y2 in range(y, y+height):
            for x2 in range(x, x+width):
                self.vals_[y2, x2] += val / float(normalization_factor)
                self.counts_[y2, x2] += 1

    def add_val_segment(self, val, segment_id, segment_map, \
                        area_normalization = True):
        """
        Add val on the segment_id \in {0, ..., num_segments-1}.
        segment_map is a numpy.int32 of the same size as the heatmap, where
        the segment_map[i, j] value is the segment-id for that (i, j)-location
        """
        area_region = 0
        for y2 in range(self.vals_.shape[0]):
            for x2 in range(self.vals_.shape[1]):
                if segment_map[y2, x2] == segment_id:
                    area_region += 1
        for y2 in range(self.vals_.shape[0]):
            for x2 in range(self.vals_.shape[1]):
                if segment_map[y2, x2] == segment_id:
                    self.vals_[y2, x2] += val / float(area_region)
                    self.counts_[y2, x2] += 1

    def normalize_counts(self):
        """
        Normalize the values by the counts, and set the counts to ones.
        """
        self.vals_ /= self.counts_
        self.counts_ = np.ones(self.counts_.shape, np.int32)

    def get_values(self):
        """
        Returns a ndarray.float64
        """
        return self.vals_

    def export_to_jpeg(self):
        """
        Returns a string of bytes containing a Jpeg visualization of the values.
        All the values < 0 are mapped to zero, and all the ones > 1.0 are
        mapped to 255. The values in between are linearly scaled.
        """
        raw_image = np.zeros(self.vals_.shape, np.float64)
        for y in range(self.vals_.shape[0]):
            for x in range(self.vals_.shape[1]):
                if self.vals_[y,x] > 1.0:
                    raw_image[y,x] = 1.0
                elif self.vals_[y,x] < 0.0:
                    raw_image[y,x] = 0.0
                else:
                    raw_image[y,x] = self.vals_[y,x]
        # TODO xxxxxxxxxxxxxxxxxxxx
        jpeg_image = StringIO.StringIO()

    def save_to_jpeg(self, filename):
        raise NotImplementedError()

    @staticmethod
    def sum_heatmaps(heatmaps):
        """
        Returns a Heatmap object produced by summing up the input list of
        heatmaps (i.e. it sums the values and the counts).
        """
        assert isinstance(heatmaps, list)
        assert len(heatmaps) > 0
        heat_shape = heatmaps[0].shape
        heat_out = Heatmap(heat_shape)
        for heat in heatmaps:
            heat_out.vals_ += heat.vals_
            heat_out.counts_ += heat.counts_
        return heat_out


#=============================================================================
