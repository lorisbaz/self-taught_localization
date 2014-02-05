import numpy as np
from skimage import segmentation
from scipy import io

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
    Extract a set of segmentations using Felzenszwalb method
    """

    def __init__(self, scales = [], sigmas = [], min_sizes = [], params = []):
        """
        Segmentation parameters for the Felzenszwalb algorithm.
        params, if specified, is a list of tuples (scale, sigma, min)
        """
        self.params_ = []
        for sg in sigmas:
            for m in min_sizes:
                for sc in scales:
                    self.params_.append( (sc, sg, m) )
        self.params_.extend(params)

    def extract(self, image):
        """
        Performs segmentation and returns a set of nd.array
        """    
        # Init the list of segmentations
        segmentations = [] 
        for param in self.params_:
            sc, sg, m = param
            segm_mask = segmentation.felzenszwalb(image, sc, sg, m) 
            segmentations.append(segm_mask)              
        return segmentations
        
#=============================================================================
class ImgSegmFromMatFiles(ImgSegm):
    """
    Load a set of segmentations stored in a certain directory
    """

    def __init__(self, directory, img_root_dir, file_list, num_levels=4):
        """
        Segmentation files stored in directory
	- directory: where segmentation files are stored
	- img_root_dir: where images are stored
	- file_list: the list of images we have
	- num_levels: number of levels for each parameter
        """
	self.directory_ = directory
	self.img_root_dir_ = img_root_dir
	self.file_list_ = file_list
	self.iterator_ = 0 # index the file_list
	self.num_levels_ = num_levels

    def extract(self, image):
        """
        Load segmentation, parse the mat files and returns a set of nd.array
        """    
	# Load file
	matfile = self.file_list_[self.iterator_][1]
	matfile = matfile.replace(self.img_root_dir_,'')
	matfile = matfile.replace('JPEG','mat')
	segm_mat = io.loadmat(self.directory_ + matfile)
	
	# Parse segmentation files
	segm_L1 = segm_mat.get('blobIndIm')
	segm_tree = segm_mat.get('tree')	
	
	# Create segmentations from the tree of labels
	segmentations = [] 
	for i in range(np.shape(segm_L1)[1]):
	    # append first-level segmentation 
            segmentations.append(segm_L1[0,i])
	    # get useful info
	    segm_mask_Li = segm_L1[0,i]
	    leaves = segm_tree[0,i]['leaves'][0][0] # crazy indexing...
	    nodes = segm_tree[0,i]['nodes'][0][0]   # believe or not,
	    maxid_segm_Li = np.max(segm_L1[0,i])    # it is correct
	    last_segm = np.max(nodes)
	    # create the other segmentations
	    while(np.shape(nodes)[0]>0):

		js = [] # support to delete nodes
		segm_mask_Li1 = np.array(segm_mask_Li) # copy 
		for j in range(np.shape(nodes)[0]):
		    if (nodes[j,1]<=maxid_segm_Li and \
			nodes[j,2]<=maxid_segm_Li):
			# merge blobs
			# new segment has label -> nodes[j,0]
			
			js.append(j)	
			num_new_segm_Li1 += 1

		# remove already-visited nodes
		np.delete(nodes, js, axis=0)
		maxid_segm_Li = maxid_segm_Li + num_new_segm_Li1
		# store segmentation 
		# TODOO: store only a subsample of levels!!!
		segmentations.append(segm_mask_Li1) 

	    #segmentations.append(segm_mask)
        
	self.iterator += 1
	return segmentations
 


