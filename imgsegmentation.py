import numpy as np
from skimage import segmentation
from scipy import io
from util import *
import skimage.io
import scipy.spatial.distance as dist
import scipy.cluster.hierarchy as hierarchy
import tempfile
import logging
import os
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

    *********************************************************
    ************** DEPRECATED *******************************
    *********************************************************
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

    How to use this class: for each image
    1) call "set_segm_name", passing the
       usual dataset key (i.e. the image name with the .JPEG extension), 
    2) call extract passing the image. Note that if segm_type_load=='warped',
       the warp the bboxes and segments according the size of this image.
    """

    def __init__(self, directory, img_root_dir, segm_type_load='original',\
                    min_sz_segm=30, subset_par=False):
        """
        Segmentation files stored in directory
	    - directory: where segmentation files are stored
	    - img_root_dir: where images are stored
        - segm_type_load: 
                'original' just the normal segments
                'warped' the coordinates of the segments are resized so 
                         to form a square.
        - min_sz_segm: min size of the bbox sorrounding the segment
        - subset_par: if True take a subset of segmentations paramters
                      to speed up the obfuscation part

        The segments have been computed from the original images 
        resized to have max edge 600 pixels.
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

        RETURNS:
        - a list of dictionaries {'bbox': BBox, 'mask': int nd.array}
          The bbox is the outer rectable enclosing the segment.
          The mask contains only 0, 1 values, and it is relative to the bbox.
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
                tmp = segm_mask[i][0]['rect'][0][0][0] 
                if (tmp[3]-tmp[1]-1 >= self.min_sz_segm_) or \
                    (tmp[2]-tmp[0]-1 >= self.min_sz_segm_): # filter small
                    # note: rect is [ymin,xmin,ymax,xmax]
                    bbox = BBox(tmp[1]-1, tmp[0]-1, tmp[3], tmp[2]) 
                    segm_now = {'bbox': bbox, \
                                'mask': segm_mask[i][0]['mask'][0][0]}
                    if self.segm_type_=='warped':
                        segm_now = self.warp_segment_(segm_now, \
                                                    orig_sz, warped_sz) 
                    if segm_now['mask']!=[]:
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
        # might happen that the size is 0 when the segments are very thin
        if segm['bbox'].ymax-segm['bbox'].ymin-1>0 and \
                            segm['bbox'].xmax-segm['bbox'].xmin-1>0: 
            segm['mask'] = np.ceil(skimage.transform.resize(segm['mask'], \
                               (segm['bbox'].ymax-segm['bbox'].ymin-1, \
                               segm['bbox'].xmax-segm['bbox'].xmin-1)))
        else:
            segm['mask'] = []
        return segm



#=============================================================================
class ImgSegm_SelSearch_Wrap(ImgSegm):
    """
    This is a wrapper that runs the selective search matlab function to 
    extract the segments.
    """

    def __init__(self, ss_version = 'fast', min_sz_segm = 20):
        """
        - ss_version:
            'fast' (default): uses a reduced set of sel search parameters
            'quality': uses all the parameters (for more info see IJCV paper)
        - min_sz_segm: min size of the bbox sorrounding the segment
        """
        self.ss_version_ = ss_version 
        self.min_sz_segm_ = min_sz_segm

    def extract(self, image):
        """
        Compute segmentation using matlab function, parse the mat files 
        and returns a set of nd.array

        RETURNS:
        - a list of dictionaries {'bbox': BBox, 'mask': int nd.array}
          The bbox is the outer rectable enclosing the segment.
          The mask contains only 0, 1 values, and it is relative to the bbox.
        """    
	    # Print
        logging.info('Running MATLAB selective search.')  
        # dump the images of the AnnotatedImages to temporary files 
        (fd, img_temp_file) = tempfile.mkstemp(suffix = '.bmp')
        os.close(fd)
        img = skimage.io.imsave(img_temp_file, image)
        # create temporary files for the .mat files
        (fd, mat_temp_file) = tempfile.mkstemp(suffix = '.mat')
        os.close(fd)
        # run the Selective Search Matlab wrapper
        img_temp_files_cell = '{\'' + img_temp_file + '\'}'
        mat_temp_files_cell = '{\'' + mat_temp_file + '\'}'
        matlab_cmd = 'selective_search_obfuscation({0}, {1}, \'{2}\')'\
                        .format(img_temp_files_cell, mat_temp_files_cell, \
                                self.ss_version_)
        command = "matlab -nojvm -nodesktop -r \"try; " + matlab_cmd + \
                "; catch; exit; end; exit\""
        logging.info('Executing command ' + command)
        if os.system(command) != 0:
            logging.error('Matlab SS script did not exit successfully!')
            return []
        # load the .mat file
        try:
            segm_mat = scipy.io.loadmat(mat_temp_file)
        except:
            logging.error('Exception while loading ' + mat_temp_file)
        # delete all the temporary files
        os.remove(img_temp_file)
        os.remove(mat_temp_file)
        # Parse segmentation files:  [0,s][i][0][X] X = 'mask', 'rect', 'size' 
        segm_blobs = segm_mat.get('hBlobs')	 
        # make segm_blobs more "usable" and filter small segments
        segm_all_list = []
        for s in range(np.shape(segm_blobs)[1]): # for each segm mask
            segm_mask = segm_blobs[0,s]
            segm_all = []
            for i in range(len(segm_mask)-1): # for each segment (last = full)
                # usual crazy/tricky indexing of the loadmat
                tmp = segm_mask[i][0]['rect'][0][0][0] 
                if (tmp[3]-tmp[1]-1 >= self.min_sz_segm_) or \
                    (tmp[2]-tmp[0]-1 >= self.min_sz_segm_): # filter small
                    # note: rect is [ymin,xmin,ymax,xmax]
                    bbox = BBox(tmp[1]-1, tmp[0]-1, tmp[3], tmp[2]) 
                    segm_now = {'bbox': bbox, \
                                'mask': segm_mask[i][0]['mask'][0][0]}
                    if segm_now['mask']!=[]:
                        segm_all.append(segm_now)
            segm_all_list.append(segm_all)
        return segm_all_list


#=============================================================================
class ImgSegm_ObfuscationSearch(ImgSegm):
    """
    This is a wrapper that runs the selective search matlab function to 
    extract the segments.
    """

    def __init__(self, network, ss_version = 'fast', min_sz_segm = 0, \
                        topC = 0, alpha = 1):
        """
        - network: neural net used for classification        
        - ss_version:
            'fast' (default): uses a reduced set of sel search parameters
            'quality': uses all the parameters (for more info see IJCV paper)
        - min_sz_segm: min size of the bbox sorrounding the segment
        - topC: keep the topC classifier result. =0 means that take the max
                response of the classifier
        - alpha: merge two distance matrices (space and confidence)
        """ 
        self.ss_version_ = ss_version
        self.min_sz_segm_ = min_sz_segm
        self.net_ = network
        self.topC_ = topC
        self.alpha_ = alpha

    def extract(self, image):
        """
        Compute segmentation using matlab function, parse the mat files 
        perform segment merging accordingly to the classification score
        and returns a list of dictionaries

        RETURNS:
        - a list of dictionaries {'bbox': BBox, 'mask': int nd.array, 
                                  'confidence': int}
          The bbox is the outer rectable enclosing the segment.
          The mask contains only 0, 1 values, and it is relative to the bbox.
          The conficende is max(1 - classification accuracy of obfuscation)
        """    
	    # Print
        logging.info('Running MATLAB selective search.')  
        # dump the images of the AnnotatedImages to temporary files 
        (fd, img_temp_file) = tempfile.mkstemp(suffix = '.bmp')
        os.close(fd)
        if len(np.shape(image))==2: # gray img
            img = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
            img[:,:,0] = image
            img[:,:,1] = image
            img[:,:,2] = image
            image = img
        img = skimage.io.imsave(img_temp_file, image)
        # create temporary files for the .mat files
        (fd, mat_temp_file) = tempfile.mkstemp(suffix = '.mat')
        os.close(fd)
        # run the Selective Search Matlab wrapper
        img_temp_files_cell = '{\'' + img_temp_file + '\'}'
        mat_temp_files_cell = '{\'' + mat_temp_file + '\'}'
        matlab_cmd = 'selective_search_obfuscation({0}, {1}, \'{2}\')'\
                        .format(img_temp_files_cell, mat_temp_files_cell, \
                                self.ss_version_)
        command = "matlab -nojvm -nodesktop -r \"try; " + matlab_cmd + \
                "; catch; exit; end; exit\""
        logging.info('Executing command ' + command)
        if os.system(command) != 0:
            logging.error('Matlab SS script did not exit successfully!')
            return []
        # load the .mat file
        try:
            segm_mat = scipy.io.loadmat(mat_temp_file)
        except:
            logging.error('Exception while loading ' + mat_temp_file)
        # delete all the temporary files
        #os.remove(img_temp_file)
        #os.remove(mat_temp_file)
        # Load only first-level segmentation (i.e., Felzenswalb) 
        segm_masks = segm_mat.get('blobIndIm')
        # classify full img
        caffe_rep_full = self.net_.evaluate(image)
        class_guess = np.argmax(caffe_rep_full)
        # make segm_blobs more "usable" and filter small segments
        segm_all_list = []
        for s in range(np.shape(segm_masks)[1]): # for each segm mask
            segm_mask = segm_masks[0,s]
            segm_all = []
            segm_ids = np.unique(segm_mask)
            max_segm_id = np.max(segm_ids)
            confidence = np.zeros(len(segm_ids))
            feature_vec = []
            logging.info('segm_mask {0} / {1} ({2} segments)'.format( \
                         s, np.shape(segm_masks)[1], max_segm_id)) 
            # compute obfuscation score for each segment
            #heatmap_tmp = np.zeros((np.shape(image)[0], np.shape(image)[1]))
            for id_segment in segm_ids: # for each segment of level 0
                # compute bbox (for filtering)
                mask = segm_mask==id_segment
                ys = np.argwhere(np.sum(mask, axis = 1) > 0)
                xs = np.argwhere(np.sum(mask, axis = 0) > 0)
                ymin = np.min(ys)
                ymax = np.max(ys)
                xmin = np.min(xs)
                xmax = np.max(xs)
                if (xmax-xmin >= self.min_sz_segm_) and \
                        (ymax-ymin >= self.min_sz_segm_): # filter small
                    image_obf = np.copy(image) # copy array
                    # obfuscation 
                    if np.shape(image.shape)[0]>2: # RGB images
                        image_obf[segm_mask==id_segment,0] = \
                                             self.net_.get_mean_img()[0]
                        image_obf[segm_mask==id_segment,1] = \
                                             self.net_.get_mean_img()[1]
                        image_obf[segm_mask==id_segment,2] = \
                                             self.net_.get_mean_img()[2]   
                    else: # GRAY images
                        image_obf[segm_mask==id_segment] = \
                                       np.mean(self.net_.get_mean_img())
                    # predict CNN reponse for obfuscation
                    caffe_rep_obf = self.net_.evaluate(image_obf)
                    # Given the class of the image, select the confidence
                    if self.topC_ == 0:
                        confidence = caffe_rep_full[class_guess] - \
                                        caffe_rep_obf[class_guess]
                    else: ### TODOOOOOOOOOOOOOOOOO ###
                        raise NotImplementedError()
                    #heatmap_tmp[mask] = confidence
                    # Build output (bbox and mask)
                    feature_vec.append([xmin+(xmax-xmin)/2.0, \
                                        ymin+(ymax-ymin)/2.0, \
                                        confidence, id_segment]) 
                    bbox = BBox(xmin, ymin, xmax, ymax, confidence)
                    mask_tmp = mask[ymin:ymax,xmin:xmax]
                    segm_all.append({'bbox': bbox, 'mask': mask_tmp})
            # Merging segments by confidence [TODOO]
            logging.info(' - Hierarchical Clustering')
            X = np.array(feature_vec)[:,0:3] # remove id segm
            X[:,0] = X[:,0]/np.shape(image)[1] # normalize
            X[:,1] = X[:,1]/np.shape(image)[0]
            D = dist.pdist(X[:,0:2], 'euclidean') + self.alpha_ * \
                dist.pdist(X[:,2].reshape((np.shape(X)[0],1)), 'euclidean')
            Z = hierarchy.linkage(D, method='average')
#            # visualize
#            import pylab as pl
#            pl.subplot(131)
#            pl.imshow(segm_mask, interpolation='nearest')
#            labs = []
#            for pos in feature_vec:
#                x,y,c,id = pos
#                pl.text(x, y, str(id), fontdict={'size': 18})
#                labs.append(str(id))
#            pl.subplot(132)
#            pl.imshow(heatmap_tmp, interpolation='nearest')
#            labs = []
#            for pos in feature_vec:
#                x,y,c,id = pos
#                pl.scatter(x,y)
#                pl.text(x, y, str(id), fontdict={'size': 18})
#                labs.append(str(id))
#            pl.subplot(133)
#            W = hierarchy.dendrogram(Z, labels = labs)
#            pl.show()
            ZZ = segm_all
            n = np.shape(Z)[0]
            segm_mask_support = np.copy(segm_mask)
            id_segments = np.array(feature_vec)[:,3].tolist()
            for i in range(n): 
                # Extract the bbox
                id1, id2, conf, num = Z[i,:]
                id1 = np.int16(id1)
                id2 = np.int16(id2)
                xmin = min(ZZ[id1]['bbox'].xmin, ZZ[id2]['bbox'].xmin)
                ymin = min(ZZ[id1]['bbox'].ymin, ZZ[id2]['bbox'].ymin)
                xmax = max(ZZ[id1]['bbox'].xmax, ZZ[id2]['bbox'].xmax)
                ymax = max(ZZ[id1]['bbox'].ymax, ZZ[id2]['bbox'].ymax)
                conf = (ZZ[id1]['bbox'].confidence + \
                            ZZ[id2]['bbox'].confidence)/2.0
                bbox = BBox(xmin, ymin, xmax, ymax, conf)
                # Extract the mask
                id_segment1 = id_segments[id1]
                id_segment2 = id_segments[id2]                 
                max_segm_id = max_segm_id + 1   
                segm_mask_support[segm_mask_support==id_segment1] = max_segm_id
                segm_mask_support[segm_mask_support==id_segment2] = max_segm_id
                mask = np.copy(segm_mask_support == max_segm_id)
                mask_tmp = mask[ymin:ymax,xmin:xmax] # [TODOOOO]
                ZZ.append({'bbox': bbox, 'mask': mask_tmp})
                id_segments.append(max_segm_id)
#                print (id1, id2, len(ZZ)-1)
#                W = hierarchy.dendrogram(Z)
#                pl.show() 
            segm_all_list.append(ZZ)
        return segm_all_list

    @staticmethod
    def segments_to_bboxes(self, segments):
        bboxes = []
        for s in np.shape(segments)[0]:
            for w in np.shape(segments[s])[0]:
                bboxes.append(segments[s][w]['bbox'])
        return bboxes 

