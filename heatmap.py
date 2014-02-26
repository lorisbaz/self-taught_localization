import numpy as np
import pylab as plt
import skimage.io
from bbox import *

class Heatmap:
    """class for Heatmap"""

    def __init__(self, width, height):
        """
        width and height are two integers.
        """
        self.vals_ = np.zeros((height, width), np.float64)
        self.counts_ = np.zeros((height, width), np.int32)
        self.segment_map_ = None
        self.description_ = ''

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

    def set_segment_map(self, segment_map):
        """
        Set the current segmentation map to use.
        segment_map can be:
        - numpy.int32 of the same size as the heatmap, where
          the segment_map[i, j] value is the segment-id for that (i, j)-location
        - list of num_segments lists, where the i-th list contains
          the (x, y) locations for the i-th segment.
        """
        if isinstance(segment_map, np.ndarray):
            # calculate the number of segments present in this map, and
            # create the segment_map_, as a list of lists
            num_segments = np.max(segment_map) + 1
            assert num_segments > 0
            # create the segment_map_, as a list of lists
            self.segment_map_ = [[] for i in range(num_segments)]
            for y in range(segment_map.shape[0]):
                for x in range(segment_map.shape[1]):
                    self.segment_map_[segment_map[y, x]].append( (x, y) )
        elif isinstance(segment_map, list):
            self.segment_map_ = segment_map
        else:
            raise ValueError('segment_map type not supported')

    def add_val_segment(self, val, segment_id, area_normalization = True):
        """
        Add val on the segment_id \in {0, ..., num_segments-1}, using the
        current segmentation map.
        """
        assert self.segment_map_ != None
        normalization_factor = 1.0
        if area_normalization:
            normalization_factor = float(len(self.segment_map_[segment_id])) \
				   / float(np.prod(np.shape(self.vals_)))
        for (x, y) in self.segment_map_[segment_id]:
            self.vals_[y, x] += val / float(normalization_factor)
            self.counts_[y, x] += 1

    def add_val_segment_mask(self, val, box, mask, \
                             area_normalization = True):
        """
        Add val on the region in box with binary mask, using the
        current segmentation map.
        """
        # extract crop
        vals_crop = np.copy(self.vals_[box.ymin:box.ymax,\
                                       box.xmin:box.xmax])
        counts_crop = np.copy(self.counts_[box.ymin:box.ymax,\
                                           box.xmin:box.xmax])                                         
        normalization_factor = 1.0
        if area_normalization:
            normalization_factor = float(np.sum(mask)) \
				                   / float(np.prod(np.shape(self.vals_)))
        # add values
        vals_crop[mask==1] += val / float(normalization_factor)
        counts_crop[mask==1] += 1
        # restore crop
        self.vals_[box.ymin:box.ymax,\
                   box.xmin:box.xmax] = vals_crop
        self.counts_[box.ymin:box.ymax,\
                     box.xmin:box.xmax] = counts_crop 

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

    def get_description(self):
        """
        Returns a string containing info about how the map has been extracted.
        """
        return self.description_

    def set_description(self, descr):
        """
        Add user-defined description.
        """
        self.description_ = descr

    def resize(self, width_new, height_new):
        """
        Resize the heatmap
        """
        self.vals_ = np.float64(skimage.transform.resize(self.vals_, \
                                (height_new, width_new)))
        self.counts_ = np.uint32(skimage.transform.resize(self.counts_, \
                                (height_new, width_new)))
        return self

    def export_to_image(self, colormap = False, factor = 1.0):
        """
        Returns a ndarray, which consists of a visualization of the values.
        All the values < 0 are mapped to zero, and all the ones > 1.0 are
        mapped to 255. The values in between are linearly scaled.
        """
        raw_image = np.zeros(self.vals_.shape, np.uint8)
        for y in range(self.vals_.shape[0]):
            for x in range(self.vals_.shape[1]):
       		tmp_image = self.vals_[y,x] * factor         
		tmp_image = round(tmp_image * 255.0)
                tmp_image = min(tmp_image, 255)	
                raw_image[y,x] = max(tmp_image, 0)
	if colormap:
	    cmap = plt.get_cmap('jet') # retrieve color map
	    raw_image = cmap(raw_image)
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
