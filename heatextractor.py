import numpy as np
from heatmap import *
from imgsegmentation import *
import logging
from bbox import *

#=============================================================================

class HeatmapExtractor:
    """class for HeatmapExtractor"""

    def __init__(self, network):
        raise NotImplementedError()

    def extract(self, image, label = ''):
    	"""
        Returns a set of heatmaps (Heatmap objects)
        """
        raise NotImplementedError()

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
        return lab_id, caffe_rep_full, idx_top_c, lab_list, num_top_c, top_accuracies

#=============================================================================

class HeatmapExtractorSegm_List(HeatmapExtractor):
    """
    Extracts a set of heatmaps built from the results of the CNN by obfuscating 
    segments of the image, considering all the saved maps of Selective search 
    algorithm. NOTE: works only with saved mat segmentation masks.
    """
    def __init__(self, network, segment, confidence_tech = 'full_obf', \
                 area_normalization = True, image_transform = 'original', \
                 num_pred = 0, quantile_pred=1.0, min_num_pred=0):
        """
        segment is of type ImgSegm.
        confidence_tech is the type of extracted confidence which can be:
        - 'only_obf': 1 - classification_score for the given label of the 
                      obfuscated image
        - 'full_obf': classification_score for the image 
                          - classification_score of the obfuscated image
        - 'full_obf_positive': max(full_obf, 0)
        area_normalization: normalize by area of the segment
        image_transform: 'original' or 'warping'
        num_pred: take the best num_pred and build the heatmaps
                    if 0, we keep only the label provided in the extract method 
                    (which clearly has to be given). In this case, quantile_pred
                    is ignored.
        quantile_pred: keep only the number of labels whose sum of the scores
                         is >= quantile_pred. Remember that the max value is 1.0
                         Note that this option can be used in combination
                         with num_pred (which can be used to cap 
                         the number of labels).
        min_num_pred: the minimum number of labels to keep.
                      if num_pred==0, this parameter is ignored
        """
        self.network_ = network
        self.segment_ = segment
        self.area_normalization_ = area_normalization
        self.confidence_tech_ = confidence_tech
        self.image_trans_ = image_transform
        self.num_pred_ = num_pred
        self.quantile_pred_ = quantile_pred
        self.min_num_pred = min_num_pred
        
    def extract(self, image, label = ''):
        """
        Perform segmentation-based obfuscation and returns a set of heatmaps 
        (Heatmap objects)
        """
        # Init the list of heatmaps and warp the image (if requested)
        heatmaps = []
        if (self.image_trans_=='warped'):
            # resize image with the same size of the CNN input
            image = skimage.transform.resize(image, \
                (self.network_.get_input_dim(), self.network_.get_input_dim()))
            image = skimage.img_as_ubyte(image)
        # select the most useful classes
        lab_id, caffe_rep_full, idx_top_c, lab_list, num_top_c, top_accuracies = \
                                    self.select_top_labels_(image, label)
        # Perform segmentation
        segm_masks = self.segment_.extract(image) # list of segmentation
        for s in range(np.shape(segm_masks)[0]): # for each segm. mask
            # init heatmap
            heatmap = []
            for i in range(num_top_c):
                heatmap.append(Heatmap(image.shape[1], \
                                        image.shape[0]))
            segm_list = segm_masks[s] # retrieve s-th mask 
            #heatmap.set_segment_map(segm_mask)
            #segm_list = np.unique(segm_mask)
            num_segments = len(segm_list) + 1
            #max_segments = np.max(segm_list) + 1
            logging.info('segm_mask {0} / {1} ({2} segments)'.format( \
                         s, len(segm_masks), num_segments))
            # obfuscation & heatmap
            for segment_i in segm_list:
                #if id_segment % (max_segments / 10) == 0:
                #    logging.info('segment {0} / {1}'.format(id_segment, \
                #                 max_segments))
                image_obf = np.copy(image) # copy array            
                # obfuscation
                box = segment_i['bbox'] 
                mask = segment_i['mask']
                image_crop = np.copy(image_obf[box.ymin:box.ymax,\
                                               box.xmin:box.xmax])
                if np.shape(image.shape)[0]>2: # RGB images
                    image_crop[mask==1,0] = self.network_.get_mean_img()[0]
                    image_crop[mask==1,1] = self.network_.get_mean_img()[1]
                    image_crop[mask==1,2] = self.network_.get_mean_img()[2]   
                else: # GRAY images
                    image_crop[mask==1] = np.mean(self.network_.get_mean_img())
                image_obf[box.ymin:box.ymax, \
                          box.xmin:box.xmax] = np.copy(image_crop)
                # predict CNN reponse for obfuscation
                caffe_rep_obf = self.network_.evaluate(image_obf)
                # Given the class of the image, select the confidence
                if self.confidence_tech_ == 'only_obf':
                    confidence = 1-caffe_rep_obf[idx_top_c]
                elif self.confidence_tech_ == 'full_obf':
                    confidence = caffe_rep_full[idx_top_c] - \
                                 caffe_rep_obf[idx_top_c]
                elif self.confidence_tech_ == 'full_obf_positive':
                    confidence = caffe_rep_full[idx_top_c] - \
                                 caffe_rep_obf[idx_top_c]
                    for i in range(num_top_c):
                        confidence[i] = max(confidence[i], 0.0)
                # update the heatmap
                for i in range(num_top_c):
                    heatmap[i].add_val_segment_mask(confidence[i], box, \
                                               mask, self.area_normalization_)
            if self.num_pred_>0:              
                for i in range(num_top_c):
                    assert len(heatmap) == num_top_c
                    heatmap[i].set_description('Computed with segmentation ' + \
                                    'obfuscation, map with {0} segments' + \
                                    ' in the {1} confidence setup. Total' + \
                                    ' of {2} maps.'.format(num_segments, \
                                    self.confidence_tech_, len(segm_masks)))
                    heatmap[i].normalize_counts()
            else:
                assert len(heatmap) == 1
                heatmap[0].set_description('Computed with segmentation ' + \
                                    'obfuscation, map with {0} segments' + \
                                    ' in the {1} confidence setup. Total' + \
                                    ' of {2} maps.'.format(num_segments, \
                                    self.confidence_tech_, len(segm_masks)))
                heatmap[0].normalize_counts()
            heatmaps.append(heatmap) # append the heatmap to the list
        if self.num_pred_>0:
            return heatmaps, lab_list, top_accuracies
        else:     
            return heatmaps        
        
#=============================================================================

class HeatmapExtractorSliding(HeatmapExtractor):
    """
    Extracts a set of heatmaps built from the results of the CNN by sliding 
    rectangular regions
    """
    def __init__(self, network, params, confidence_tech = 'single_win', \
                 area_normalization = True, num_pred = 0, quantile_pred=1.0,\
                 min_num_pred=0):
        """
      	network is of type Network
        params are tuples of sliding window parameters:
        - bbox_sz: size of the bounding box
        - stride: regular stride of the windows over the image
        confidence_tech is the type of extracted confidence which can be:
        - 'single_win': 1 - classification_score for the given label of the 
                      slide window
        area_normalization: normalize by area of the segment
        num_pred: take the best num_pred and build the heatmaps
                  if 0, we keep only the label provided in the extract method 
                  (which clearly has to be given). In this case, quantile_pred
                    is ignored.
        quantile_pred: keep only the number of labels whose sum of the scores
                         is >= quantile_pred. Remember that the max value is 1.0
                         Note that this option can be used in combination
                         with num_pred (which can be used to cap 
                         the number of labels).
        min_num_pred: the minimum number of labels to keep.
                      if num_pred==0, this parameter is ignored
        """
        self.network_ = network
        self.params_ = params
        self.area_normalization_ = area_normalization
        self.confidence_tech_ = confidence_tech
        self.num_pred_ = num_pred
        self.quantile_pred_ = quantile_pred
        self.min_num_pred = min_num_pred
  
    def extract(self, image, label = ''):
        """
        Compute CNN response for sliding windows and returns a set of heatmaps 
            (Heatmap objects)
        """
        # select the most useful classes
        lab_id, caffe_rep_full, idx_top_c, lab_list, num_top_c, top_accuracies = \
                                    self.select_top_labels_(image, label)
        # Init the list of heatmaps
        heatmaps = []
        # Cycle over boxes        
        for param in self.params_: # for box parameter
            box_sz, stride = param
            # init heatmap
            heatmap = []
            for i in range(num_top_c):
                heatmap.append(Heatmap(image.shape[1], \
                                       image.shape[0]))
            # generate indexes
            xs = np.linspace(0, image.shape[1]-box_sz, \
                             (image.shape[1]-box_sz)/float(stride)+1)
            xs = np.int32(xs)
            ys = np.linspace(0, image.shape[0]-box_sz, \
                             (image.shape[0]-box_sz)/float(stride)+1)
            ys = np.int32(ys)
            logging.info('sliding window {0} / {1} ({2} windows) '\
                         .format(np.shape(heatmaps)[0]+1, \
    				     len(self.params_), len(xs)*len(ys))) 
            # crop img and compute CNN response
            for x in xs:
                for y in ys:
                    # predict CNN reponse for current window
                    caffe_rep_win = \
                         self.network_.evaluate(image[y:y+box_sz, x:x+box_sz]) 
                    # Given the class of the image, select the confidence
                    confidence = caffe_rep_win[idx_top_c]
                    for i in range(num_top_c):
                        confidence[i] = max(confidence[i], 0.0)
                        heatmap[i].add_val_rect(confidence[i], x, y, \
                                box_sz, box_sz, self.area_normalization_)
            if self.num_pred_>0:
                assert len(heatmap) == num_top_c
                for i in range(num_top_c):
                    heatmap[i].set_description('Computed with sliding win ' + \
                                    'approach, with window size {0} and' + \
                                    ' stride {1}. Total' + \
                                    ' of {2} maps.'.format(box_sz, \
                                    stride, len(self.params_)))
                    heatmap[i].normalize_counts()
            else:
                assert len(heatmap) == 1
                heatmap[0].set_description('Computed with sliding win ' + \
                                    'approach, with window size {0} and' + \
                                    ' stride {0}. Total' + \
                                    ' of {2} maps.'.format(box_sz, \
                                    stride, len(self.params_)))
                heatmap[0].normalize_counts()
            heatmaps.append(heatmap) # append the heatmap to the list 
        if self.num_pred_>0:
            return heatmaps, lab_list, top_accuracies
        else:     
            return heatmaps

#=============================================================================

class HeatmapExtractorBox(HeatmapExtractor):
    """
    Extracts a set of heatmaps built from the results of the CNN by obfuscating 
    rectangular regions
    """
    def __init__(self, network, params, confidence_tech = 'full_obf', \
                 area_normalization = True, num_pred = 0, quantile_pred=1.0,
                 min_num_pred=0):
        """
        network is of type Network
	    params are tuples of sliding window parameters:
	    bbox_sz: size of the bounding box
	    stride: regular stride of the windows over the image
        confidence_tech is the type of extracted confidence which can be:
          - 'only_obf': 1 - classification_score for the given label of the 
                      obfuscated image
          - 'full_obf': classification_score for the image 
                          - classification_score of the obfuscated image
          - 'll_obf_positive': max(full_obf, 0)
        area_normalization: normalize by area of the segment
        num_pred: take the best num_pred and build the heatmaps
                    if 0, we keep only the label provided in the extract method 
                    (which clearly has to be given). In this case, quantile_pred
                    is ignored.
        quantile_pred: keep only the number of labels whose sum of the scores
                         is >= quantile_pred. Remember that the max value is 1.0
                         Note that this option can be used in combination
                         with num_pred (which can be used to cap 
                         the number of labels).
        min_num_pred: the minimum number of labels to keep.
                      if num_pred==0, this parameter is ignored
        """
        self.network_ = network
        self.params_ = params
        self.area_normalization_ = area_normalization
        self.confidence_tech_ = confidence_tech
        self.num_pred_ = num_pred 
        self.quantile_pred_ = quantile_pred
        self.min_num_pred = min_num_pred
  
    def extract(self, image, label = ''):
        """
        Perform box-based obfuscation and returns a set of heatmaps 
        (Heatmap objects)
        """
        # Init the list of heatmaps
        heatmaps = []
        # resize image with the same size of the CNN input
        image_resz = skimage.transform.resize(image, \
             (self.network_.get_input_dim(), self.network_.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz) 
        # select the most useful classes
        lab_id, caffe_rep_full, idx_top_c, lab_list, num_top_c, top_accuracies = \
                                    self.select_top_labels_(image, label)
        # Cycle over boxes        
        for param in self.params_: # for box parameter
            box_sz, stride = param
            logging.info('box mask {0} / {1} '\
                          .format(np.shape(heatmaps)[0]+1, \
    				      len(self.params_)))
            # init heatmap
            heatmap = []
            for i in range(num_top_c):
                heatmap.append(Heatmap(image_resz.shape[1], \
                                       image_resz.shape[0]))
            # generate indexes (resized img)
            xs = np.linspace(0, image_resz.shape[1]-box_sz, \
                             (image_resz.shape[1]-box_sz)/float(stride)+1)
            xs = np.int32(xs)
            ys = np.linspace(0, image_resz.shape[0]-box_sz, \
                             (image_resz.shape[0]-box_sz)/float(stride)+1)
            ys = np.int32(ys) 
            # obfuscation & heatmap for each box
            for x in xs:
                for y in ys:
                    image_obf = np.copy(image_resz) # copy array            
                    # obfuscation 
                    if np.shape(image_resz.shape)[0]>2: # RGB images
                        image_obf[y:y+box_sz,x:x+box_sz,0] = \
                                                 self.network_.get_mean_img()[0]
                        image_obf[y:y+box_sz,x:x+box_sz,1] = \
                                                 self.network_.get_mean_img()[1]
                        image_obf[y:y+box_sz,x:x+box_sz,2] = \
                                                 self.network_.get_mean_img()[2]   
                    else: # GRAY images
                        image_obf[y:y+box_sz,x:x+box_sz] = \
                                           np.mean(self.network_.get_mean_img())
                    # predict CNN reponse for obfuscation
                    caffe_rep_obf = self.network_.evaluate(image_obf)
                    # Given the class of the image, select the confidence
                    if self.confidence_tech_ == 'only_obf':
                        confidence = 1-caffe_rep_obf[idx_top_c]
                    elif self.confidence_tech_ == 'full_obf':
                        confidence = caffe_rep_full[idx_top_c] - \
                                    caffe_rep_obf[idx_top_c]
                    elif self.confidence_tech_ == 'full_obf_positive':
                        confidence = caffe_rep_full[idx_top_c] - \
                                    caffe_rep_obf[idx_top_c]
                        for i in range(num_top_c):
                            confidence[i] = max(confidence[i], 0.0)
                    # update the heatmap
                    for i in range(num_top_c):
                        heatmap[i].add_val_rect(confidence[i], x, y, \
                                box_sz, box_sz, self.area_normalization_)
            if self.num_pred_>0:  
                assert len(heatmap) == num_top_c
                for i in range(num_top_c):
                    heatmap[i].set_description('computed with gray box' + \
                                    'obfuscation, with window size {0}' + \
                                    ' and stride {1}. Total of {2} maps.'\
                                    .format(box_sz, stride, len(self.params_)))
                    heatmap[i].normalize_counts()
            else:
                assert len(heatmap) == 1
                heatmap[0].set_description('computed with gray box' + \
                                    'obfuscation, with window size {0}' + \
                                    ' and stride {1}. Total of {2} maps.'\
                                    .format(box_sz, stride, len(self.params_)))
                heatmap[0].normalize_counts()
            heatmaps.append(heatmap) # append the heatmap to the list
        if self.num_pred_>0:
            return heatmaps, lab_list, top_accuracies
        else:     
            return heatmaps
