import numpy as np
from skimage import segmentation
from scipy import io
from util import *
import skimage.io
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hierarchy
import tempfile
import logging
import os
from bbox import *

class SelfTaughtLoc:
    """class for Self Taught Localizer"""

    def __init__(self):
        raise NotImplementedError()

    def extract(self):
        """
        Returns a set of segments
        """
        raise NotImplementedError()


#==============================================================================
class SelfTaughtLoc_Grayout(SelfTaughtLoc):
    """
    This is a wrapper that runs the segmentation method from a  matlab function
    and generates the segments by agglomerative clustering.
    Paper version without any modifications. Kept for back-compatibility.
    """

    def __init__(self, network, img_segmenter, min_sz_segm = 0, \
                        topC = 0, alpha = np.ones((4,)), obfuscate_bbox = False, \
                        function_stl = 'diversity', padding = 0.0, \
                        layer = 'fc7'):
        """
        - network: neural net used for classification
        - img_segmenter: segmentation algorithm in the class ImgSegm
        - min_sz_segm: min size of the bbox sorrounding the segment
        - topC: keep the topC classifier result. =0 means that take the max
                response of the classifier
        - alpha: merge two distance matrices (space and confidence)
        - obfuscate_bbox: if True, obfuscates the bbox surrounding the segment
                if False (default), obfuscates the segment
        - function_stl: select the function to use for the grayout term
                'diversity' (default): drop_segment_1 - drop_segment_2
                'similarity': 1 - max((1-drop_segment_1), (1-drop_segment_2)) *
                              |(1-drop_segment_1) - (1-drop_segment_2)|
                'similarity+cnnfeature': same as similarity + an additional term
                    that considers the similarity between segments using the CNN
                    feature representation ('fc7')
        - padding: only if cnnfeature is used. It is the percentage
                espressed between 0.0 and 1.0 of the bbox that is enlaged to
                contain the context surrounding it
        - layer: only if cnnfeature is used. The layer of the cnn used by the
                similarity function.
        """
        self.img_segmenter_ = img_segmenter
        self.min_sz_segm_ = min_sz_segm
        self.net_ = network
        self.topC_ = topC
        self.alpha_ = alpha
        self.obfuscate_bbox_ = obfuscate_bbox
        self.function_stl_ = function_stl
        self.padding_ = padding
        self.layer_ = layer

    @staticmethod
    def segments_to_bboxes(self, segments):
        bboxes = []
        for s in np.shape(segments)[0]:
            for w in np.shape(segments[s])[0]:
                bboxes.append(segments[s][w]['bbox'])
        return bboxes


    def extract_greedy(self, image, label=''):
        """
        Compute segmentation using matlab function, parse the mat files
        perform segment merging accordingly to the obfuscation score and
        other similarities. It returns a list of dictionaries.
        The GT label can also be provided as input if available.

        RETURNS:
        - a list of dictionaries {'bbox': BBox, 'mask': int nd.array,
                                  'confidence': int}
          The bbox is the outer rectable enclosing the segment.
          The mask contains only 0, 1 values, and it is relative to the bbox.
          The conficende is max(1 - classification accuracy of obfuscation)
        """
        # segment image
        segm_masks = self.img_segmenter_.extract(image)
        # classify full img
        caffe_rep_full = self.net_.evaluate(image)
        if label=='':
            class_guess = np.argmax(caffe_rep_full)
        else: # if the label is provided, we use the GT!
            class_guess = self.net_.get_label_id(label)
            self.topC_ = 0 # hack to force the use of the class_guess
        # Our Obfuscation Search---
        segm_all_list = []
        image_sz = np.shape(image)[0:2]
        for s in range(np.shape(segm_masks)[1]): # for each segm mask
            segm_mask = np.uint16(segm_masks[0,s])
            segm_all = []
            segm_ids = np.unique(segm_mask)
            max_segm_id = np.max(segm_ids)
            confidence = np.zeros(len(segm_ids))
            logging.info('segm_mask {0} / {1} ({2} segments)'.format( \
                         s + 1, np.shape(segm_masks)[1], max_segm_id))
            # Generate the basic segments
            for id_segment in segm_ids: # for each segment of level 0
                # compute bbox (for filtering)
                mask = segm_mask==id_segment
                ys = np.argwhere(np.sum(mask, axis = 1) > 0)
                xs = np.argwhere(np.sum(mask, axis = 0) > 0)
                ymin = np.min(ys)
                ymax = np.max(ys)
                xmin = np.min(xs)
                xmax = np.max(xs)
                if (xmax-xmin >= self.min_sz_segm_) and \
                        (ymax-ymin >= self.min_sz_segm_): # filter small
                    # compute confidence
                    confidence = self.obfuscation_confidence_(image, \
                         segm_mask, id_segment, caffe_rep_full, class_guess, \
                         xmin, xmax, ymin, ymax)
                    # compute CNN features
                    feature_vec = self.extract_bbox_cnnfeature_(image, \
                                                    xmin, xmax, ymin, ymax)
                    # Build the list of segments with positive confidence
                    mask_tmp = np.copy(mask[ymin:ymax,xmin:xmax])
                    bbox = BBox(xmin, ymin, xmax, ymax, max(confidence, 0.0))
                    segm_all.append({'bbox': bbox, 'mask': mask_tmp, \
                                     'id': id_segment, 'feature': feature_vec})
            # Init the similarity matrix (contains only neighbouring pairs)
            similarity = self.compute_similarity_sets_(segm_all, segm_all, \
                                                       self.alpha_, image_sz)
            similarity = self.zero_diag_values_(similarity)
            # Clustering
            S = list(segm_all)
            segm_mask_supp = np.copy(segm_mask)
            while len(S)!=1:
                # Find highest similarity
                i_max, j_max = np.where(similarity == np.max(similarity))
                i_max = i_max[0]
                j_max = j_max[0]
                # Merge regionsi
                max_segm_id += 1
                segment, segm_mask_supp = self.merge_segments_(S[i_max], \
                                    S[j_max], image, segm_mask_supp, \
                                    max_segm_id, caffe_rep_full, class_guess)
                # Remove Similarity i_max and jmax:
                # Note: order of max and min is to preserve the structure
                similarity = np.delete(similarity, max(i_max, j_max), 0)
                similarity = np.delete(similarity, max(i_max, j_max), 1)
                dummy = S.pop(max(i_max, j_max))
                similarity = np.delete(similarity, min(i_max, j_max), 0)
                similarity = np.delete(similarity, min(i_max, j_max), 1)
                dummy = S.pop(min(i_max, j_max))
                # Add merged segment to S
                S.append(segment)
                assert np.sum(segment['mask'])>0
                # compute similarity of the new segment with the rest
                simtmp = np.zeros((np.shape(similarity)[0] + 1, \
                                    np.shape(similarity)[1] + 1))
                simtmp[0:np.shape(similarity)[0], \
                            0:np.shape(similarity)[1]] = similarity.copy()
                simtmp[:,-1] = self.compute_similarity_sets_([segment], S,\
                                                     self.alpha_, image_sz)
                simtmp[-1,:] = simtmp[:,-1]
                similarity = simtmp.copy()
                similarity = self.zero_diag_values_(similarity)
                # Add merged segment to the output
                segm_all.append(segment)
            segm_all_list.append(segm_all)

        return segm_all_list


    def zero_diag_values_(self, similarity):
        """
        Set to zero the elements on the diagonal (self-similarity) to avoit
        "self-merging". Similarity must be a square matrix.
        """
        for i in range(np.shape(similarity)[0]):
            similarity[i,i] = 0

        return similarity


    def merge_segments_(self, segm_1, segm_2, image, segm_mask_support, \
                        max_segm_id, caffe_rep_full, class_guess):
        """
        Merge two segments in one, and update the segmentation mask replacing
        the old segments with the new one. It also computes the obfuscation
        score for the new segment.
        """
        # Merge bboxes
        xmin = min(segm_1['bbox'].xmin, segm_2['bbox'].xmin)
        ymin = min(segm_1['bbox'].ymin, segm_2['bbox'].ymin)
        xmax = max(segm_1['bbox'].xmax, segm_2['bbox'].xmax)
        ymax = max(segm_1['bbox'].ymax, segm_2['bbox'].ymax)
        # Extract the mask
        id_segment1 = segm_1['id']
        id_segment2 = segm_2['id']
        segm_mask_support[segm_mask_support==id_segment1] = max_segm_id
        segm_mask_support[segm_mask_support==id_segment2] = max_segm_id
        mask = np.copy(segm_mask_support == max_segm_id)
        mask_tmp = np.copy(mask[ymin:ymax,xmin:xmax])
        # perform obfuscation
        conf = self.obfuscation_confidence_(image, segm_mask_support, \
                                    max_segm_id, caffe_rep_full, class_guess,\
                                    xmin, xmax, ymin, ymax)
        # create bbox object
        bbox = BBox(xmin, ymin, xmax, ymax, conf)
        # compute CNN features
        feature_vec = self.extract_bbox_cnnfeature_(image, xmin, xmax, \
                                                    ymin, ymax)
        # save output structure
        segment = {'bbox': bbox, 'mask': mask_tmp, 'id': max_segm_id, \
                   'feature': feature_vec}

        return segment, segm_mask_support


    def obfuscation_confidence_(self, image, segm_mask, id_segment, \
                                caffe_rep_full, class_guess, \
                                xmin, xmax, ymin, ymax):
        """
        Compute the obfuscation score for a given segment with label id_segment
        """
        image_obf = np.copy(image) # copy array
        # If ON, instead of using the segment, we obfuscate the bbox
        # surrouding the segment
        if self.obfuscate_bbox_:
            if np.shape(image.shape)[0]>2: # RGB images
                image_obf[ymin:ymax, xmin:xmax, 0] = \
                                self.net_.get_mean_img()[0]
                image_obf[ymin:ymax, xmin:xmax, 1] = \
                                self.net_.get_mean_img()[1]
                image_obf[ymin:ymax, xmin:xmax, 2] = \
                                self.net_.get_mean_img()[2]
            else: # GRAY images
                image_obf[ymin:ymax, xmin:xmax] = \
                                np.mean(self.net_.get_mean_img())
        else: # If OFF, use segments
            # obfuscation
            if np.shape(image.shape)[0]>2: # RGB images
                image_obf[segm_mask==id_segment,0] = \
                                 self.net_.get_mean_img()[0]
                image_obf[segm_mask==id_segment,1] = \
                                 self.net_.get_mean_img()[1]
                image_obf[segm_mask==id_segment,2] = \
                                 self.net_.get_mean_img()[2]
            else: # GRAY images
                image_obf[segm_mask==id_segment] = \
                           np.mean(self.net_.get_mean_img())
        # predict CNN reponse for obfuscation
        caffe_rep_obf = self.net_.evaluate(image_obf)
        # Given the class of the image, select the confidence
        if self.topC_ == 0:
            confidence = max(caffe_rep_full[class_guess] - \
                            caffe_rep_obf[class_guess], 0.0)
        else:
            idx_sort = np.argsort(caffe_rep_full)[::-1]
            idx_sort = idx_sort[0:self.topC_]
            confidence = max(np.sum(caffe_rep_full[idx_sort] - \
                                    caffe_rep_obf[idx_sort]), 0.0)
        return confidence

    def extract_bbox_cnnfeature_(self, image, xmin, xmax, ymin, ymax):
        """
        Compute the cnn feature vector for a given bbox. If padding>0, the bbox
        is enlarged to include the context.
        """
        if 'cnnfeature' in self.function_stl_:
            if self.padding_ > 0.0:
                offsetx = (xmax - xmin)*self.padding_
                offsety = (ymax - ymin)*self.padding_
                ymin -= offsety
                ymax += offsety
                xmin -= offsetx
                xmax += offsetx
                # check that they are still inside the img
                img_height, img_width = np.shape(image)[0:2]
                if ymin < 0:
                    ymin = 0
                if xmin < 0:
                    xmin = 0
                if ymax > img_height:
                    ymax = img_height
                if xmax > img_width:
                    xman = img_width
            # crop image
            image_box = np.copy(image[ymin:ymax, xmin:xmax])
            # predict CNN reponse for obfuscation
            caffe_rep = self.net_.evaluate(image_box, layer_name = self.layer_)
            # select the feature layer
            return caffe_rep
        else:
            return 0.0

    def compute_similarity_sets_(self, segm_set1, segm_set2, alpha, image_sz):
        """
        Compute the similarity matrix between two segment sets, using the
        obfuscation, the fill and the size metrics. They are combined by the
        ndarray alpha of size (4,).
        """
        img_area = np.float(np.prod(image_sz))
        similarity = np.zeros((len(segm_set1), len(segm_set2)))
        for i in range(len(segm_set1)):
            segm_i = segm_set1[i]
            for j in range(len(segm_set2)): # only lower triang
                segm_j = segm_set2[j]
                if True: # self.adjacency_test_(segm_i, segm_j):
                    # Compute new obfuscation score (eulidean dist)
                    if self.function_stl_=='diversity':
                        s_obf = np.linalg.norm(segm_i['bbox'].confidence-\
                                            segm_j['bbox'].confidence)
                    elif 'similarity' in self.function_stl_:
                        s_obf = 1 - max(1-segm_i['bbox'].confidence, \
                                        1-segm_j['bbox'].confidence) * \
                                        abs((1-segm_i['bbox'].confidence)-\
                                            (1-segm_j['bbox'].confidence))
                    # Compute size measure
                    s_size = 1 - (np.sum(segm_i['mask']) + \
                                    np.sum(segm_j['mask']))/img_area
                    # Compute fill measure
                    size_BB = (max(segm_i['bbox'].xmax, segm_j['bbox'].xmax) -\
                              min(segm_i['bbox'].xmin, segm_j['bbox'].xmin)) *\
                              (max(segm_i['bbox'].ymax, segm_j['bbox'].ymax) -\
                              min(segm_i['bbox'].ymin, segm_j['bbox'].ymin))
                    s_fill = 1 - (size_BB - np.sum(segm_i['mask']) -
                                    np.sum(segm_j['mask']))/img_area
                    # Merge measures
                    similarity[i,j] = alpha[0]*s_obf + alpha[1]*s_size \
                                        + alpha[2]*s_fill
                    if 'cnnfeature' in self.function_stl_:
                        s_feature = compare_feature_vec(\
                                        segm_i['feature'], segm_j['feature'])
                        similarity[i,j] += alpha[3]*s_feature
        return similarity

#   def nms(self):
#       """
#       Perform Non-maximum suppression. [TODO] now it is done in the compute
#       statistics class.
#       """
#
#       raise NotImplementedError()

    def adjacency_test_(self, segm_1, segm_2):
        """
        Establish if two segments are adjacent. [TODO] speed up computation by
        comparing only few neighboor elements.
        """

        raise NotImplementedError()
