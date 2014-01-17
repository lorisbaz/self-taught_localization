

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
        self.sigma   = [0.1,0.3,0.6,1]
        self.min     = 40
        self.scale   = [100,300,500]

    def extract(self, image, label = ''):
	"""
	Returns a set of heatmaps (Heatmap objects)
	"""
        # retrieve the label id
        lab_id      = self.network.get_label_id(label)
        
        # Init the list of heamaps
        heamaps     = []
        
        # segmentation of the image 
        #if np.shape(image.shape)[0]>2: 
        #    multich=True
        #else:
        #    multich=False
        #segm_mask   = segmentation.slic(image, n_segments=100, compactness=10.0, max_iter=10, sigma=None, spacing=None, multichannel=multich, convert2lab=True, ratio=None)    
        segm_mask   = segmentation.felzenszwalb(image, self.sigma, self.min, self.scale) # [felzenszwalb & Huttenlocher, IJCV 2004]
        ## guess the output of the network of the whole image (not needed)
        #caffe_rep   = self.network.eval(image)
             
        # obfuscation & heatmap
        heatmap     = Heatmap() # init heatmaps     
        heatmap_loc = np.zeros(np.shape(segm_mask))
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
            heatmap_loc[segm_mask==id_segment] = 1-caffe_rep_obf[lab_id]    
            
        heatmap.set(heatmap_loc) # set the heatmaps 

        heamaps.append(heatmap) # append the heatmap to the list
        
        
#=============================================================================

# TODO here to implement sliding windows

#=============================================================================

# TODO here to implement Gray box obfuscation
