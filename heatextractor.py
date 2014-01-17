

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
    """
    def __init__(self, network):
        self.network = network

    def extract(self, image, label = ''):
	"""
	Returns a set of heatmaps (Heatmap objects)
	"""
        # segmentation of the image 
        #segm_mask   = segmentation.felzenszwalb(image, segm_sigma=0.4, segm_min=40, scale=300) # [felzenszwalb & Huttenlocher, IJCV 2004]
        if np.shape(image.shape)[0]>2: 
            multich=True
        else:
            multich=False
        segm_mask   = segmentation.slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=None, spacing=None, multichannel=multich, convert2lab=True, ratio=None)    
        # guess the output of the network of the whole image (not needed)
        caffe_rep   = self.network.eval(image)
        
        # retrieve the label id
        lab_id      = self.network.get_label_id(label)
        
        # obfuscation & heatmap
        heatmap = np.zeros(np.shape(segm_mask))
        for id_segment in range(np.max(segm_mask)+1):
            image_obf  = np.array(image) # copy array
            
            if np.shape(image.shape)[0]>2: 
                image_obf[segm_mask==id_segment,0] = IMAGENET_MEAN[0]
                image_obf[segm_mask==id_segment,1] = IMAGENET_MEAN[1]
                image_obf[segm_mask==id_segment,2] = IMAGENET_MEAN[2]   
            else: # consider ldg images
                image_obf[segm_mask==id_segment] = np.mean(IMAGENET_MEAN)
                
            # predict CNN reponse for obfuscation
            caffe_rep_obf   = self.network.eval(image_obf)
            
            # Given the class of the image, select the confidence
            heatmap[segm_mask==id_segment] = 1-caffe_rep_obf[lab_id]
        
        return heamap


#=============================================================================

# TODO here to implement sliding windows

#=============================================================================

# TODO here to implement Gray box obfuscation
