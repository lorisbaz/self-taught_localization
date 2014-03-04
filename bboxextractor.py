import numpy as np
import cv2
import logging
from matplotlib import pyplot
import sklearn
import sklearn.cluster
import sys

from bbox import *
from heatmap import *


class BBoxExtractor:
    def extract(self, img, heatmaps):
        """
        Returns a set of bboxes, extracted from the given image and heatmaps.
        INPUT:
        img is a ndarray;
        heatmaps is a list of ndarrays;

        OUTPUT:
        - a list of BBox
        - a list of (image, description) with the same number of elements 
               of the list of bboxes which is implementation-dependent and
               could be used for visualization/debugging purposes
        Note that the outputs might be empty lists (e.g. in case of an error).
        """
        raise NotImplementError()

    @staticmethod
    def get_bbox_from_connected_components_(mask, heatmap, object_values):
        """
        PROTECTED METHOD.
        Returns a set of BBox objects, calculated from the outer rectangles
        of the connected components of mask (4-points connectivity),
        using the 'object_values' for the
        labeling (if a pixels has value any of the object_values, then we
        declare that pixel part of the object).
        The confidence value is calculated by using the corresponding normalized
        content of the heatmap.
        INPUT:
        mask: int ndarray
        heatmap: ndarray
        object_values: list of integers
        """
        def is_object(x, obj_vals):
            for o in obj_vals:
                if x == o:
                    return True
            return False
        # make sure the input is consistent, and initialize some vars
        assert mask.shape == heatmap.shape
        visited = np.zeros(mask.shape, dtype=np.int32)
        out_bboxes = []
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if visited[y,x]:
                    continue
                visited[y,x] = 1
                if not(is_object(mask[y,x], object_values)):
                    continue
                # found a seed: (x, y) belongs to the object.
                # run DFS, while keeping track of the min/max x/y values
                # for each connected component found
                stack = [(x,y)]
                rect = [sys.maxint, sys.maxint, -sys.maxint, -sys.maxint]
                while len(stack) > 0:
                    x, y = stack.pop()
                    visited[y,x] = 1
                    if is_object(mask[y,x], object_values):
                        rect[0] = min(x, rect[0]); rect[1] = min(y, rect[1])
                        rect[2] = max(x+1, rect[2]); rect[3] = max(y+1, rect[3])
                        if x-1 >= 0 and not(visited[y,x-1]):
                            stack.append((x-1,y))
                        if x+1 < mask.shape[1] and not(visited[y,x+1]):
                            stack.append((x+1,y))
                        if y-1 >= 0 and not(visited[y-1,x]):
                            stack.append((x,y-1))
                        if y+1 < mask.shape[0] and not(visited[y+1,x]):
                            stack.append((x,y+1))
                assert rect[0] < sys.maxint
                confidence = 0.0
                for y2 in range(rect[1], rect[3]):
                    for x2 in range(rect[0], rect[2]):
                        confidence += heatmap[y2,x2]
                # the confidence is normalized to the area of the region
                confidence /= float(rect[2]-rect[0]) \
                              * float(rect[3]-rect[1])
                out_bboxes.append(BBox(rect[0], rect[1], rect[2], rect[3], \
                                       confidence))
        return out_bboxes

#=============================================================================

class GrabCutBBoxExtractor(BBoxExtractor):
    def __init__(self, min_bbox_size=0.05, grab_cut_rounds=10, \
                       consider_pr_fg=True, grab_cut_init = 'kmeans'):
        """
        min_bbox_size: the minimum (normalized) size of the outputted bboxes
        grab_cut_rounds: number of rounds for the grabcut algorithm
        consider_pr_fg: whether or not to consider as foreground also the pixels
                        labeled as "weak foreground" by the grabcut algo.
        grab_cut_init: string, indicating which method initializes the initial mask
                       for the grabcut. either 'kmeans', or 'gmm'
        """
        self.min_bbox_size_ = min_bbox_size
        self.gc_rounds_ = grab_cut_rounds
        if consider_pr_fg:
            self.gc_fg_labels_ = [cv2.GC_FGD, cv2.GC_PR_FGD]
        else:
            self.gc_fg_labels_ = [cv2.GC_FGD]
        self.grab_cut_init = grab_cut_init

    def extract(self, img, heatmaps):
        assert isinstance(img, np.ndarray)
        # NOTE: currently , we support a single heatmap
        assert isinstance(heatmaps, list)
        assert len(heatmaps) == 1
        heatmap = heatmaps[0]
        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        out_image_desc = []
        # Gray imgs have to be converted in 3D gray images
        if (len(np.shape(img))==2):
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        # 1) learn a kMeans with 4 gaussians on the heatmap values,
        #    returning the four thresholds
        if self.grab_cut_init == "kmeans":
            thresholds = self.get_thresholds_from_kmeans_(heatmap)
        elif self.grab_cut_init == "gmm":
            thresholds = self.get_thresholds_from_gmm_(heatmap)
        else:
            raise ValueError('parameter grab_cut_init not valid')
        if len(thresholds) == 0:
            return [], []
        # 2) quantize the heatmap into four segmentation labels:
        #    cv2.GC_BGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD, cv2.GC_FGD
        mask = self.get_grabcut_init_mask_(heatmap, thresholds)
        mask_image = self.get_image_from_mask_(img, mask)
        out_image_desc.append( (mask_image, 'initial k-means mask') )
        # 3) run GrabCut
        gc_img = cv2.resize(img.copy(), (mask.shape[1], mask.shape[0])).copy()
        assert mask.dtype == np.uint8
        assert gc_img.dtype == np.uint8
        gc_rect = None
        gc_bgdModel = np.zeros((1,65), np.float64)
        gc_fgdModel = np.zeros((1,65), np.float64)
        gc_mask = mask.copy()
        cv2.grabCut(gc_img, gc_mask, gc_rect, gc_bgdModel, gc_fgdModel, \
                    self.gc_rounds_, mode=cv2.GC_INIT_WITH_MASK)
        assert gc_mask.shape[0] == mask.shape[0]
        assert gc_mask.shape[1] == mask.shape[1]
        mask_image = self.get_image_from_mask_(img, gc_mask)
        out_image_desc.append( (mask_image, 'grab-cut mask') )
        # 4) extractor the bbox, as the outer rectangles of the final segments
        bboxes = BBoxExtractor.get_bbox_from_connected_components_( \
                              gc_mask, heatmap, self.gc_fg_labels_)
        # 5) normalize the bboxes to one
        for bbox in bboxes:
            bbox.normalize_to_outer_box(BBox(0,0,mask.shape[1],mask.shape[0]))
        # 6) remove very small bboxes, and normalize the bboxes to one
        out_bboxes = []
        for bbox in bboxes:
            if bbox.area() > self.min_bbox_size_:
                out_bboxes.append(bbox)
        return out_bboxes, out_image_desc


    def get_image_from_mask_(self, img, mask):
        out = cv2.resize(img.copy(), (mask.shape[1], mask.shape[0])).copy()
        for y in range(mask.shape[0]):
            for x in range(mask.shape[1]):
                if mask[y,x] == cv2.GC_BGD:
                    out[y,x,:] *= 0
                elif mask[y,x] == cv2.GC_FGD:
                    out[y,x,0] = 0
                    out[y,x,1] = 255
                    out[y,x,2] = 0
                elif mask[y,x] == cv2.GC_PR_BGD:
                    out[y,x,0] = 0
                    out[y,x,1] = 0
                    out[y,x,2] = 255
                elif mask[y,x] == cv2.GC_PR_FGD:
                    out[y,x,0] = 255
                    out[y,x,1] = 0
                    out[y,x,2] = 0
                else:
                    assert 0
        return out

    def get_thresholds_from_kmeans_(self, heatmap):
        """
        Learn a kMeans with 4 clusters, and return the thresholds that separate
        the clusters: [m01, m12, m23]
        Returns [] in case of error (e.g. when heatmap is all black)
        """
        g = sklearn.cluster.KMeans(n_clusters = 4)
        data = heatmap.reshape((heatmap.size, 1))
        g.fit(data)
        assert g.cluster_centers_.shape[0] == 4
        assert g.cluster_centers_.shape[1] == 1
        centers = np.sort(g.cluster_centers_.copy(), axis=None)
        if not(centers[0] < centers[1] < centers[2] < centers[3]):
            logging.warning('Kmeans failed with the collisions of some centroids.')
            return []
        m01 = (centers[0]+centers[1]) / 2.0
        m12 = (centers[1]+centers[2]) / 2.0
        m23 = (centers[2]+centers[3]) / 2.0
        return [m01, m12, m23]

    def get_thresholds_from_gmm_(self, heatmap):
        """
        Learn a GMM with 4 gaussians, and return the thresholds that separate
        the posterior values: [m01, m12, m23].
        Returns [] in case of error (e.g. when heatmap is all black)
        """
        # TODooooooooooooooooooooooooooooooooooooooooo
        raise NotImplementError()
        g = sklearn.cluster.KMeans(n_clusters = 4)
        data = heatmap.reshape((heatmap.size, 1))
        g.fit(data)
        assert g.cluster_centers_.shape[0] == 4
        assert g.cluster_centers_.shape[1] == 1
        centers = np.sort(g.cluster_centers_.copy(), axis=None)
        if not (centers[0] < centers[1] < centers[2] < centers[3]):
            return []
        m01 = (centers[0]+centers[1]) / 2.0
        m12 = (centers[1]+centers[2]) / 2.0
        m23 = (centers[2]+centers[3]) / 2.0
        return [m01, m12, m23]

    def get_grabcut_init_mask_(self, heatmap, thresholds):
        """
        quantize the heatmap into four segmentation labels:
        cv2.GC_BGD, cv2.GC_FGD, cv2.GC_PR_BGD, cv2.GC_PR_FGD.
        It returns a np. given the tresholds
        """
        m01, m12, m23 = thresholds
        assert m01 < m12 < m23
        mask = np.zeros(heatmap.shape, np.uint8)
        data = heatmap
        for y in range(data.shape[0]):
            for x in range(data.shape[1]):
                if data[y,x] < m01:
                    mask[y,x] = cv2.GC_BGD
                elif data[y,x] < m12:
                    mask[y,x] = cv2.GC_PR_BGD
                elif data[y,x] < m23:
                    mask[y,x] = cv2.GC_PR_FGD
                else:
                    mask[y,x] = cv2.GC_FGD
        return mask
