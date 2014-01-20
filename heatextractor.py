import numpy as np
from skimage import segmentation
from Heatmap import *


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
    def __init__(self, network, sigmas, mins, scales):
        self.network = network
        # segmentation parameters
        self.sigma_   = sigmas
        self.min_     = mins
        self.scale_   = scales

    def extract(self, image, label = ''):
	"""
	Perform segmentation-based obfuscationa and returns a set of heatmaps (Heatmap objects)
	"""
        # retrieve the label id
        lab_id      = self.network.get_label_id(label)
        
        # Init the list of heamaps
        heamaps     = []
        
        for sg in range(np.shape(self.sigma_)[0]):
            for m in range(np.shape(self.min_)[0]):
                for sc in range(np.shape(self.scale_)[0]):
        
                    heatmap     = Heatmap() # init heatmap     

                    # segmentation of the image 
                    #if np.shape(image.shape)[0]>2: 
                    #    multich=True
                    #else:
                    #    multich=False
                    #segm_mask   = segmentation.slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=None, spacing=None, multichannel=multich, convert2lab=True, ratio=None)    
                    segm_mask   = segmentation.felzenszwalb(image, self.sigma_[sg], self.min_[m], self.scale_[sc]) # [felzenszwalb & Huttenlocher, IJCV 2004]
                    ## guess the output of the network of the whole image (not needed)
                    #caffe_rep   = self.network.eval(image)

                    # obfuscation & heatmap
                    heatmap_loc = np.zeros(np.shape(segm_mask))
                    for id_segment in range(np.max(segm_mask)):
                        image_obf  = np.array(image) # copy array

                        if np.shape(image.shape)[0]>2: 
                            image_obf[segm_mask==id_segment,0] = IMAGENET_MEAN[0]
                            image_obf[segm_mask==id_segment,1] = IMAGENET_MEAN[1]
                            image_obf[segm_mask==id_segment,2] = IMAGENET_MEAN[2]   
                        else: # consider ldg images
                            image_obf[segm_mask==id_segment] = np.mean(IMAGENET_MEAN)

                        # predict CNN reponse for obfuscation
                        caffe_rep_obf   = self.network.evaluate(image_obf)

                        # Given the class of the image, select the confidence
                        heatmap_loc[segm_mask==id_segment] = 1-caffe_rep_obf[lab_id]    

                    heatmap.set(heatmap_loc) # set the heatmaps 

                    heamaps.append(heatmap) # append the heatmap to the list
                    
        return heamaps
        
        
#=============================================================================

# TODO here to implement sliding windows

#=============================================================================

# TODO here to implement Gray box obfuscation
