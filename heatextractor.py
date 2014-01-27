import numpy as np
from heatmap import *
from imgsegmentation import *

class HeatmapExtractor:
    """class for HeatmapExtractor"""

    def __init__(self, network):
        raise NotImplementedError()

    def extract(self, image, label = ''):
	"""
	Returns a set of heatmaps (Heatmap objects)
	"""
        raise NotImplementedError()


#=============================================================================

class HeatmapExtractorSegm(HeatmapExtractor):
    """
    Extracts a set of heatmaps built from the results of the CNN by obfuscating 
    segments of the image
    """
    def __init__(self, network, segment, confidence_tech = 'full_obf', \
                 area_normalization = True):
        self.network_ = network
        self.segment_ = segment
        self.area_normalization_ = area_normalization
        self.confidence_tech_ = confidence_tech
        
    def extract(self, image, label = ''):
        """
        Perform segmentation-based obfuscation and returns a set of heatmaps 
        (Heatmap objects)
        """
        # retrieve the label id
        lab_id = self.network_.get_label_id(label)    
        # Init the list of heamaps
        heamaps = []
        # Classify the full image without obfuscation
        if (self.confidence_tech_ == 'full_obf') or \
           (self.confidence_tech_ == 'full_obf_positive'):
            caffe_rep_full = self.network_.evaluate(image)
        # Perform segmentation
        segm_masks = self.segment_.extract(image) # list of segmentation masks    
        for s in range(np.shape(segm_masks)[0]): # for each segm. mask
            heatmap = Heatmap(image.shape[1], image.shape[0]) # init heatmap     
            segm_mask = segm_masks[s] # retrieve s-th mask
            heatmap.set_segment_map(segm_mask)
            # obfuscation & heatmap
            for id_segment in range(np.max(segm_mask)+1):
                image_obf = np.array(image) # copy array            
                # obfuscation 
                if np.shape(image.shape)[0]>2: # RGB images
                    image_obf[segm_mask==id_segment,0] = \
                                                self.network_.get_mean_img()[0]
                    image_obf[segm_mask==id_segment,1] = \
                                                self.network_.get_mean_img()[1]
                    image_obf[segm_mask==id_segment,2] = \
                                                self.network_.get_mean_img()[2]   
                else: # GRAY images
                    image_obf[segm_mask==id_segment] = \
                                           np.mean(self.network_.get_mean_img())
                # predict CNN reponse for obfuscation
                caffe_rep_obf = self.network_.evaluate(image_obf)
                # Given the class of the image, select the confidence
                if self.confidence_tech_ == 'only_obf':
                    confidence = 1-caffe_rep_obf[lab_id]
                elif self.confidence_tech_ == 'full_obf':
                    confidence = caffe_rep_full[lab_id] - caffe_rep_obf[lab_id]
                elif self.confidence_tech_ == 'full_obf_positive':
                    confidence = max(caffe_rep_full[lab_id] - \
                                     caffe_rep_obf[lab_id], 0.0)
                # update the heatmap
                heatmap.add_val_segment(confidence, id_segment, segm_mask, \
                                        self.area_normalization_) 
            heamaps.append(heatmap) # append the heatmap to the list                    
        return heamaps
        
        
#=============================================================================

class HeatmapExtractorSliding(HeatmapExtractor):
    """
    Extracts a set of heatmaps built from the results of the CNN by obfuscating 
    rectangular regions
    """
    def __init__(self, network, box_sz, stride, confidence_tech = 'full_obf', \
                 area_normalization = True):
        self.network_ = network
        self.box_sz_ = box_sz
        self.stride_ = stride
        self.area_normalization_ = area_normalization
        self.confidence_tech_ = confidence_tech
        
    def extract(self, image, label = ''):
	"""
	Compute CNN response for sliding windows and returns a set of heatmaps 
        (Heatmap objects)
	"""
        # retrieve the label id
        lab_id = self.network_.get_label_id(label)    
        # Init the list of heamaps
        heamaps = []
        # Classify the full image without obfuscation
        if (self.confidence_tech_ == 'full_obf') or \
           (self.confidence_tech_ == 'full_obf_positive'):
            caffe_rep_full = self.network_.evaluate(image)
        # Cycle over boxes        
        for box_sz in self.box_sz_: # for box parameter
            heatmap = Heatmap(image.shape[1], image.shape[0]) # init heatmap
            # generate indexes
            xs = np.linspace(0, image.shape[1]-box_sz, \
                             (image.shape[1]-box_sz)/float(self.stride_)+1)
            xs = np.int32(xs)
            ys = np.linspace(0, image.shape[1]-box_sz, \
                             (image.shape[0]-box_sz)/float(self.stride_)+1)
            ys = np.int32(ys)
            # crop img and compute CNN response
            for x in xs:
                for y in ys:
                    # predict CNN reponse for current window
                    if np.shape(image.shape)[0]>2: # RGB images
                        caffe_rep_win = \
                          self.network_.evaluate(image[y:y+box_sz,x:x+box_sz,:]) 
                    else: # GRAY images
                        caffe_rep_win = \
                          self.network_.evaluate(image[y:y+box_sz,x:x+box_sz])
                    # Given the class of the image, select the confidence
                    if self.confidence_tech_ == 'only_obf':
                        confidence = 1-caffe_rep_win[lab_id]
                    elif self.confidence_tech_ == 'full_obf':
                        confidence = caffe_rep_full[lab_id] - \
                                     caffe_rep_win[lab_id]
                    elif self.confidence_tech_ == 'full_obf_positive':
                        confidence = max(caffe_rep_full[lab_id] - \
                                         caffe_rep_win[lab_id], 0.0)
                    # update the heatmap
                    heatmap.add_val_rect(confidence, x, y, box_sz, box_sz, \
                                         self.area_normalization_) 
                    #print str(x) + ' ' + str(y) + ' __ '
            heatmap.normalize_counts()
            heamaps.append(heatmap) # append the heatmap to the list                    
        return heamaps


#=============================================================================

class HeatmapExtractorBox(HeatmapExtractor):
    """
    Extracts a set of heatmaps built from the results of the CNN by obfuscating 
    rectangular regions
    """
    def __init__(self, network, box_sz, stride, confidence_tech = 'full_obf', \
                 area_normalization = True):
        self.network_ = network
        self.box_sz_ = box_sz
        self.stride_ = stride
        self.area_normalization_ = area_normalization
        self.confidence_tech_ = confidence_tech
        
    def extract(self, image, label = ''):
	"""
	Perform box-based obfuscation and returns a set of heatmaps 
        (Heatmap objects)
	"""
        # retrieve the label id
        lab_id = self.network_.get_label_id(label)    
        # Init the list of heamaps
        heamaps = []
        # Classify the full image without obfuscation
        if (self.confidence_tech_ == 'full_obf') or \
           (self.confidence_tech_ == 'full_obf_positive'):
            caffe_rep_full = self.network_.evaluate(image)
        # Cycle over boxes        
        for box_sz in self.box_sz_: # for box parameter
            heatmap = Heatmap(image.shape[1], image.shape[0]) # init heatmap
            # generate indexes
            xs = np.linspace(0, image.shape[1]-box_sz, \
                             (image.shape[1]-box_sz)/float(self.stride_)+1)
            xs = np.int32(xs)
            ys = np.linspace(0, image.shape[0]-box_sz, \
                             (image.shape[0]-box_sz)/float(self.stride_)+1)
            ys = np.int32(ys)
            # obfuscation & heatmap for each box
            for x in xs:
                for y in ys:
                    image_obf = np.array(image) # copy array            
                    # obfuscation 
                    if np.shape(image.shape)[0]>2: # RGB images
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
                        confidence = 1-caffe_rep_obf[lab_id]
                    elif self.confidence_tech_ == 'full_obf':
                        confidence = caffe_rep_full[lab_id] - \
                                     caffe_rep_obf[lab_id]
                    elif self.confidence_tech_ == 'full_obf_positive':
                        confidence = max(caffe_rep_full[lab_id] - \
                                         caffe_rep_obf[lab_id], 0.0)
                    # update the heatmap
                    heatmap.add_val_rect(confidence, x, y, box_sz, box_sz, \
                                         self.area_normalization_) 
                    #print str(x) + ' ' + str(y) + ' __ '
            heatmap.normalize_counts()
            heamaps.append(heatmap) # append the heatmap to the list                    
        return heamaps
