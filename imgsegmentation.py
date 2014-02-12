import numpy as np
from skimage import segmentation
from scipy import io
from util import *
import skimage.io
import logging

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

    def __init__(self, directory, img_root_dir, fix_sz, subset_par=False, num_levels=3):
        """
        Segmentation files stored in directory
	    - directory: where segmentation files are stored
	    - img_root_dir: where images are stored
	    - fix_sz: size of centered image block
        - subset_par: if True take a subset of segmentations paramters
                      to speed up the obfuscation part
        - num_levels: number of levels for each parameterv
        """
        self.directory_ = directory
        self.img_root_dir_ = img_root_dir
        self.imagename_ = None # index the file_list
        self.segmname_ = None
        self.num_levels_ = num_levels
        self.fix_sz_ = fix_sz
        self.subset_ = subset_par

    def extract(self, image):
        """
        Load segmentation, parse the mat files and returns a set of nd.array
        """    
	    # Print
        logging.info('Loading segmentations from disk')
        # Load file
        segm_mat = io.loadmat(self.directory_ + '/' + self.segmname_)
	
        # Parse segmentation files
        segm_L1 = segm_mat.get('blobIndIm')
        segm_tree = segm_mat.get('tree')	
	
        # Create segmentations from the tree of labels
        segm_all = []
        if self.subset_:
            range_segm = [1,3] # select only k=100
        else:
            range_segm = range(np.shape(segm_L1)[1])

        for i in range_segm: 
            segmentations = [] 	
            # make sure that is uint16 (I spent a day to find this bug!!)
            segm_L1[0,i] = np.uint16(segm_L1[0,i])   
	        # append first-level segmentation 
            segmentations.append(segm_L1[0,i])
	        # get useful info
            segm_mask_Li = np.copy(segm_L1[0,i])
            leaves = segm_tree[0,i]['leaves'][0][0] # crazy indexing...
            nodes = segm_tree[0,i]['nodes'][0][0]   # believe or not,
            n_segm_Li = np.max(segm_L1[0,i]) 	
            maxid_segm_Li = n_segm_Li + 1           # it is correct
            last_segm = np.max(nodes)
            # create the other segmentations
            num_new_segm_Li1 = 0	 
            while(np.shape(nodes)[0]>0):
            #while(np.max(nodes)>maxid_segm_Li): 
                js = [] # support to delete nodes
                segm_mask_Li1 = np.copy(segmentations[-1]) # copy	
                for j in range(np.shape(nodes)[0]):
                    if (nodes[j,1]<=maxid_segm_Li and \
                        nodes[j,2]<=maxid_segm_Li):
                        # merged segments have label nodes[j,0]
                        segm_mask_Li1[segm_mask_Li1==nodes[j,1]] = nodes[j,0]
                        segm_mask_Li1[segm_mask_Li1==nodes[j,2]] = nodes[j,0] 	
                        js.append(j)	
                        num_new_segm_Li1 += 1

		        # remove already-visited nodes
                segm_mask_Li = np.copy(segm_mask_Li1)
                nodes = np.delete(nodes, js, axis=0)
                maxid_segm_Li = n_segm_Li + num_new_segm_Li1
                # store segmentation 
                segmentations.append(segm_mask_Li1) 
            
            if self.subset_:
                start = 1 # start from second-level segmentation
            else:
                start = 0
            # keep num_levels segmentations (last flat segmentation removed)
            rule_last = np.shape(segmentations)[0] - 1 	  
            #segm_all.append(segmentations[0]) # always keep the first layer
            for j in range(start,rule_last + 1, \
                     np.uint16((rule_last-start)/(self.num_levels_ - 1))):
                segm_all.append(segmentations[j])
	
        #pl.subplot(6,6,1)
        #pl.imshow(image)
        #for k in range(min(np.shape(segm_all)[0],35)):
        #    pl.subplot(6,6,k+2)
        #    pl.imshow(segm_all[k],interpolation='nearest')
        #    pl.colorbar()
        #pl.show()

        # resize and crop center (like original img)	
        segm_all_rz = np.zeros((np.shape(segm_all)[0], \
                                np.shape(image)[0],np.shape(image)[1]),\
                                dtype=np.uint16)
        for k in range(np.shape(segm_all)[0]):
            # resize (not needed!)
            #factor = np.float(np.max(segm_all[k]))
            #TMP = np.copy(segm_all[k]/factor) # project in [0,1]
            #TMP = resize_image_max_size(TMP, self.fix_sz_)
            #TMP = np.uint16(crop_image_center(TMP*factor))
            #TMP = resize_mat_max_size_nn(TMP, )
            # store results
            segm_all_rz[k,:,:] = np.uint16(crop_image_center(segm_all[k]))
 
        return segm_all_rz

    def set_image_name(self, imagename):
        """
        Move the pointer to the next image
        """
        self.imagename_ = imagename

    def set_segm_name(self, imagename):
        """
        Set both image name and segmentation name to 
        synchronize the segmenter with the image list
        """
        self.imagename_ = imagename
        segmname = imagename        
        segmname = segmname.replace(self.img_root_dir_,'')
        segmname = segmname.replace('JPEG','mat') 
        self.segmname_ = segmname
 
