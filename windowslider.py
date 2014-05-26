import numpy as np
from heatmap import *
from imgsegmentation import *
import logging
from bbox import *

class WindowSlider:
    """ Perform Sliding window and return a set of bboxes with confidece"""

    def __init__(self, slide_params, network,  num_pred = 5, \
                            quantile_pred = 1.0, min_num_pred = 0.0):
        self.slide_params = slide_params
        self.num_pred_ = num_pred
        self.quantile_pred_ = quantile_pred
        self.min_num_pred = min_num_pred
        self.network_ = network

    def evaluate(self, image, label = ''):
        """
        Compute CNN response for sliding windows and returns a list
        of bboxes.
        """
        # select the most useful classes
        lab_id, caffe_rep_full, idx_top_c, \
                lab_list, num_top_c, top_accuracies = \
                                    self.select_top_labels_(image, label)
        # Init the bboxes
        bboxes_final = []
        image_width, image_height = np.shape(image)[0:2]
        # Cycle over boxes
        n_slide = 1
        for param in self.slide_params: # for box parameter
            box_sz, stride = param
            # generate indexes
            xs = np.linspace(0, image.shape[1]-box_sz, \
                             (image.shape[1]-box_sz)/float(stride)+1)
            xs = np.int32(xs)
            ys = np.linspace(0, image.shape[0]-box_sz, \
                             (image.shape[0]-box_sz)/float(stride)+1)
            ys = np.int32(ys)
            logging.info('sliding window {0} / {1} ({2} windows) '\
                         .format(n_slide, len(self.slide_params), \
                         len(xs)*len(ys)))
            # crop img and compute CNN response
            for x in xs:
                for y in ys:
                    # predict CNN reponse for current window
                    caffe_rep_win = \
                         self.network_.evaluate(image[y:y+box_sz, x:x+box_sz])
                    # Given the class of the image, select the confidence
                    confidence = caffe_rep_win[idx_top_c]
                    # Create bboxes
                    bboxes = []
                    for i in range(num_top_c):
                        confidence[i] = max(confidence[i], 0.0)
                        bbox_this = BBox(float(x), float(y), \
                                            float(x+box_sz), float(y+box_sz), \
                                            confidence[i])
                        bbox_this.normalize_to_outer_box(\
                                    BBox(0.0, 0.0, image_width, image_height))
                        bboxes.append(bbox_this)
                    bboxes_final.extend(bboxes)
            n_slide += 1
        return bboxes_final

    def select_top_labels_(self, image, label):
        # retrieve the label id
        if label == '':
            assert self.num_pred_ > 0, 'parameter label has to be specified'
            lab_id = None
        else:
            lab_id = self.network_.get_label_id(label)
        # Classify the full image without obfuscation
        caffe_rep_full = self.network_.evaluate(image)
        # select top num_pred classes
        if self.num_pred_ > 0:
            labels = self.network_.get_labels()
            assert self.num_pred_ <= len(labels)
            idx_sorted_scores = (np.argsort(caffe_rep_full)[::-1]).tolist()
            idx_top_c = []
            quantile = 0.0
            for idx in idx_sorted_scores:
                idx_top_c.append(idx)
                quantile += caffe_rep_full[idx]
                if ((quantile >= self.quantile_pred_) \
                       or (len(idx_top_c) >= self.num_pred_)) \
                    and (len(idx_top_c) > self.min_num_pred):
                    break
            if (lab_id != None) and not(lab_id in idx_top_c):
                idx_top_c.append(lab_id)
            # selection of top labels (tricky)
            lab_list = np.array(labels)[idx_top_c].tolist()
            top_accuracies = caffe_rep_full[idx_top_c]
            num_top_c = len(idx_top_c)
        else:
            idx_top_c = [lab_id]
            lab_list = [label]
            num_top_c = 1
            top_accuracies = 1.0
        return lab_id, caffe_rep_full, idx_top_c, lab_list, num_top_c, top_accuracies


