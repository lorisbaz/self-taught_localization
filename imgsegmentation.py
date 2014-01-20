import numpy as np
from skimage import segmentation

class ImgSegm:
    """class for ImageSegmentation"""

    def __init__(self):
        raise NotImplementedError()

    def extract(self):
	"""
	Returns a set of segmented images
	"""
        raise NotImplementedError()


#=============================================================================
class ImgSegmFelzen(ImgSegm):
    """
    Extract a set of segmentations depending on the selected method
    """
    def __init__(self, sigmas, mins, scales):
        # segmentation parameters
        self.sigma_ = sigmas
        self.min_ = mins
        self.scale_ = scales

    def extract(self, image):
	"""
	Performs segmentation and returns a set of nd.array
	"""    
        # Init the list of segmentations
        segmentations = [] 
        for sg in range(np.shape(self.sigma_)[0]):
            for m in range(np.shape(self.min_)[0]):
                for sc in range(np.shape(self.scale_)[0]):
                    segm_mask = segmentation.felzenszwalb(image, self.sigma_[sg],\
                                            self.min_[m], self.scale_[sc]) 
                    segmentations.append(segm_mask) # append the heatmap to the lis              
        return segmentations
        
        
#=============================================================================

# TODO here to implement slic segmentation

