import numpy as np
import skimage.io

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
        assert isinstance(segment_map, np.ndarray)
        normalization_factor = 1.0
        if area_normalization:
            area_region = 0
            for y2 in range(self.vals_.shape[0]):
                for x2 in range(self.vals_.shape[1]):
                    if segment_map[y2, x2] == segment_id:
                        area_region += 1
            normalization_factor = float(area_region)
        for y2 in range(self.vals_.shape[0]):
            for x2 in range(self.vals_.shape[1]):
                if segment_map[y2, x2] == segment_id:
                    self.vals_[y2, x2] += val / float(normalization_factor)
                    self.counts_[y2, x2] += 1

    def normalize_counts(self):
        """
        Normalize the values by the counts, and set the counts to ones.
        """
        for y in range(self.vals_.shape[0]):
            for x in range(self.vals_.shape[1]):
                if self.counts_[y,x] > 0:
                    self.vals_[y,x] /= self.counts_[y,x]
        self.counts_ = np.ones(self.counts_.shape, np.int32)

    def get_values(self):
        """
        Returns a ndarray.float64 containing the values of the heatmap.
        """
        return self.vals_

    def export_to_image(self):
        """
        Returns a ndarray, which consists of a visualization of the values.
        All the values < 0 are mapped to zero, and all the ones > 1.0 are
        mapped to 255. The values in between are linearly scaled.
        """
        raw_image = np.zeros(self.vals_.shape, np.uint8)
        for y in range(self.vals_.shape[0]):
            for x in range(self.vals_.shape[1]):
                raw_image[y,x] = round(self.vals_[y,x] * 255.0)
                raw_image[y,x] = min(raw_image[y,x], 255)
                raw_image[y,x] = max(raw_image[y,x], 0)
        return raw_image

    def save_to_image(self, filename):
        """
        Same as 'export_to_image', but saving the visualization to a file.
        """
        image = self.export_to_image()
        skimage.io.imsave(filename, image)

    @staticmethod
    def sum_heatmaps(heatmaps):
        """
        Returns a Heatmap object produced by summing up the input list of
        heatmaps (i.e. it sums the values and the counts).
        """
        assert isinstance(heatmaps, list)
        assert len(heatmaps) > 0
        heat_shape = heatmaps[0].vals_.shape
        heat_out = Heatmap(heat_shape[1], heat_shape[0])
        for heat in heatmaps:
            heat_out.vals_ += heat.vals_
            heat_out.counts_ += heat.counts_
        return heat_out

#=============================================================================
