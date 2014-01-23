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
    def __init__(self, network, segment, confidence_tech = 'full_obf', area_normalization = True):
        self.network = network
        self.segment = segment
        self.area_normalization = area_normalization
        self.confidence_tech = confidence_tech
        
    def extract(self, image, label = ''):
	"""
	Perform segmentation-based obfuscationa and returns a set of heatmaps (Heatmap objects)
	"""
        # retrieve the label id
        lab_id = self.network.get_label_id(label)    
        # Init the list of heamaps
        heamaps = []
        # Classify the full image without obfuscation
        if (self.confidence_tech == 'full_obf') or (self.confidence_tech == 'full_obf_positive'):
            caffe_rep_full = self.network.evaluate(image)
        # Perform segmentation
        segm_masks = self.segment.extract(image) # list of segmentation masks    
        for s in range(np.shape(segm_masks)[0]): # for each segm. mask
            heatmap = Heatmap(image.shape[1], image.shape[0]) # init heatmap     
            segm_mask = segm_masks[s] # retrieve s-th mask
            # obfuscation & heatmap
            for id_segment in range(np.max(segm_mask)+1):
                image_obf = np.array(image) # copy array            
                # obfuscation 
                if np.shape(image.shape)[0]>2: # RGB images
                    image_obf[segm_mask==id_segment,0] = self.network.get_mean_img()[0]
                    image_obf[segm_mask==id_segment,1] = self.network.get_mean_img()[1]
                    image_obf[segm_mask==id_segment,2] = self.network.get_mean_img()[2]   
                else: # GRAY images
                    image_obf[segm_mask==id_segment] = np.mean(self.network.get_mean_img())
                # predict CNN reponse for obfuscation
                caffe_rep_obf = self.network.evaluate(image_obf)
                # Given the class of the image, select the confidence
                if self.confidence_tech == 'only_obf':
                    confidence = 1-caffe_rep_obf[lab_id]
                elif self.confidence_tech == 'full_obf':
                    confidence = caffe_rep_full[lab_id] - caffe_rep_obf[lab_id]
                elif self.confidence_tech == 'full_obf_positive':
                    confidence = max(caffe_rep_full[lab_id] - caffe_rep_obf[lab_id], 0.0)
                # update the heatmap
                heatmap.add_val_segment(confidence, id_segment, segm_mask, self.area_normalization) 
            heamaps.append(heatmap) # append the heatmap to the list                    
        return heamaps, segm_masks
        
        
#=============================================================================

# TODO here to implement sliding windows

#=============================================================================

# TODO here to implement Gray box obfuscation
