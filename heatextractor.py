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
    def __init__(self, network, segment):
        self.network = network
        self.segment = segment
        
    def extract(self, image, label = ''):
	"""
	Perform segmentation-based obfuscationa and returns a set of heatmaps (Heatmap objects)
	"""
        # retrieve the label id
        lab_id      = self.network.get_label_id(label)
        
        # Init the list of heamaps
        heamaps     = []
        
        # Classify the full image without obfuscation
        caffe_rep_full   = net.predict(image)
        
        # Perform segmentation
        segm_masks = self.segment.extract(image) # list of segmentation masks
        
        for s in range(np.shape(segm_masks)[0]): # for each segm. mask
        
            heatmap     = Heatmap(img.shape[0], img.shape[1]) # init heatmap     

            segm_mask   = segm_masks[s] # retrieve s-th mask

            # obfuscation & heatmap
            heatmap_loc = np.zeros(np.shape(segm_mask))
            for id_segment in range(np.max(segm_mask)):
                image_obf  = np.array(image) # copy array
                
                # obfuscation 
                if np.shape(image.shape)[0]>2: 
                    image_obf[segm_mask==id_segment,0] = self.network.get_mean_img()[0]
                    image_obf[segm_mask==id_segment,1] = self.network.get_mean_img()[1]
                    image_obf[segm_mask==id_segment,2] = self.network.get_mean_img()[2]   
                else: # consider ldg images
                    image_obf[segm_mask==id_segment] = np.mean(self.network.get_mean_img())

                # predict CNN reponse for obfuscation
                caffe_rep_obf   = self.network.evaluate(image_obf)

                # Given the class of the image, select the confidence
                #confidence = 1-caffe_rep_obf[lab_id]     ## TODO: test this
                confidence = caffe_rep_full[lab_id] - caffe_rep_obf[lab_id]
                
                # update the heatmap ## TODO: test normalize vs. unnormalized
                heatmap.add_val_segment(confidence, id_segment, segm_mask) 

            heamaps.append(heatmap) # append the heatmap to the list
                    
        return heamaps
        
        
#=============================================================================

# TODO here to implement sliding windows

#=============================================================================

# TODO here to implement Gray box obfuscation
