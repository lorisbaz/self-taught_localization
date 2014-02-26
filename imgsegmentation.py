import numpy as np
from skimage import segmentation
from scipy import io
from util import *
import skimage.io
import logging
from bbox import *

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

    def __init__(self, directory, img_root_dir, subset_par=False, \
                 start_lv=1 , num_levels=3):
        """
        Segmentation files stored in directory
	    - directory: where segmentation files are stored
	    - img_root_dir: where images are stored
        - subset_par: if True take a subset of segmentations paramters
                      to speed up the obfuscation part
        - num_levels: number of levels for each parameterv
        """
        self.directory_ = directory
        self.img_root_dir_ = img_root_dir
        self.imagename_ = None # index the file_list
        self.segmname_ = None
        self.num_levels_ = num_levels
        self.subset_ = subset_par
        self.start_lv_ = start_lv

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
    
            # keep num_levels segmentations (last flat segmentation removed)
            rule_last = np.shape(segmentations)[0] - 1 	  
            tmp_start_lv = self.start_lv_
            stept = np.uint16((rule_last-tmp_start_lv)/(self.num_levels_ - 1)) 
            if stept == 0: # not enough segmentations
                tmp_start_lv = 0 
                stept = np.uint16((rule_last-tmp_start_lv)/ \
                                    (self.num_levels_ - 1)) 
            for j in range(tmp_start_lv,rule_last + 1, stept):
                segm_all.append(segmentations[j])

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

    def get_segm_name(self):
        return self.segmname_

#=============================================================================
class ImgSegmFromMatFiles_List(ImgSegm):
    """
    Load a set of segmentations stored in a certain directory and create a list
    of segmentation blobs (not segmentation maps anymore)
    """

    def __init__(self, directory, img_root_dir, segm_type_load='original',\
                    min_sz_segm=30, subset_par=False):
        """
        Segmentation files stored in directory
	    - directory: where segmentation files are stored
	    - img_root_dir: where images are stored
        - segm_type_load: either 'original' or 'warped'
        - min_sz_segm: min size of the bbox sorrounding the segment
        - subset_par: if True take a subset of segmentations paramters
                      to speed up the obfuscation part
        """
        self.directory_ = directory
        self.img_root_dir_ = img_root_dir
        self.imagename_ = None # index the file_list
        self.segmname_ = None
        self.segm_type_ = segm_type_load
        self.min_sz_segm_ = min_sz_segm
        self.subset_ = subset_par

    def extract(self, image):
        """
        Load segmentation, parse the mat files and returns a set of nd.array
        """    
	    # Print
        logging.info('Loading segmentations from disk')
        # sizes
        warped_sz = np.shape(image)[0:2]
        # Load file
        segm_mat = io.loadmat(self.directory_ + '/' + self.segmname_)
        # Parse segmentation files:  [0,s][i][0][X] X = 'mask', 'rect', 'size' 
        segm_blobs = segm_mat.get('hBlobs')	 
        # make segm_blobs more "usable" and filter small segments
        segm_all_list = []
        for s in range(np.shape(segm_blobs)[1]): # for each segm mask
            segm_mask = segm_blobs[0,s]
            orig_sz = tuple(segm_mask[-1][0]['rect'][0][0][0][2:4])
            segm_all = []
            for i in range(len(segm_mask)-1): # for each segment (last = full)
                # usual crazy/tricky indexing of the loadmat
                if np.sqrt(segm_mask[i][0]['size'][0][0][0]) \
                            >= self.min_sz_segm_:
                    tmp = segm_mask[i][0]['rect'][0][0][0]
                    # note: rect is [ymin,xmin,ymax,xmax]
                    bbox = BBox(tmp[1]-1, tmp[0]-1, tmp[3], tmp[2]) 
                    segm_now = {'bbox': bbox, \
                                'mask': segm_mask[i][0]['mask'][0][0]}
                    if self.segm_type_=='warped':
                        segm_now = self.warp_segment_(segm_now, \
                                                    orig_sz, warped_sz) 
                    segm_all.append(segm_now)
            segm_all_list.append(segm_all)
        if self.subset_: # keep only 4 segmentation maps
            segm_all_list = segm_all_list[::2]
        return segm_all_list

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

    def get_segm_name(self):
        return self.segmname_
    
    def warp_segment_(self, segm, orig_sz, warped_sz):
        """
        Resizes the segment to be consistent with the image warping (if any)
        """
        prop = (warped_sz[0]/float(orig_sz[0]), warped_sz[1]/float(orig_sz[1]))
        segm['bbox'] = BBox(np.floor(segm['bbox'].xmin * prop[1]), \
                            np.floor(segm['bbox'].ymin * prop[0]), \
                            np.floor(segm['bbox'].xmax * prop[1]), \
                            np.floor(segm['bbox'].ymax * prop[0]))
        segm['mask'] = np.ceil(skimage.transform.resize(segm['mask'], \
                               (segm['bbox'].ymax-segm['bbox'].ymin-1, \
                               segm['bbox'].xmax-segm['bbox'].xmin-1)))
        return segm
