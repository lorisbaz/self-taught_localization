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
class ImgSegmMatWraper(ImgSegm):
    """
    Extract a set of segmentations using the selective search matlab wrapper
    of the Felzenszwalb method. Note: Selective Search results are not in the
    output!
    """

    def __init__(self, ss_version = 'fast'):
        """
        Segmentation parameters for the selective search algorithm.
        """
        self.ss_version_ = ss_version

    def extract(self, image):
        """
        Performs segmentation and returns a set of nd.array
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
        matlab_cmd = 'selective_search_obfuscation_optimized({0}, {1}, \'{2}\')'\
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
        # Load only first-level segmentation (i.e., Felzenswalb)
        segm_masks = segm_mat.get('blobIndIm')

        return segm_masks


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
    *********************************************************
    ************** DEPRECATED *******************************
    *********************************************************
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
                if (tmp[3]-tmp[1]-1 >= self.min_sz_segm_) and \
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
                        topC = 0, alpha = 1, obfuscate_bbox = False, \
                        similarity = False):
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
        self.obfuscate_bbox_ = obfuscate_bbox
        self.similarity_ = similarity

    @staticmethod
    def segments_to_bboxes(self, segments):
        bboxes = []
        for s in np.shape(segments)[0]:
            for w in np.shape(segments[s])[0]:
                bboxes.append(segments[s][w]['bbox'])
        return bboxes


    def extract_greedy(self, image, label=''):
        """
        Compute segmentation using matlab function, parse the mat files
        perform segment merging accordingly to the obfuscation score and
        other similarities. It returns a list of dictionaries.
        The GT label can also be provided as input if available.

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
        os.remove(img_temp_file)
        os.remove(mat_temp_file)
        # Load only first-level segmentation (i.e., Felzenswalb)
        segm_masks = segm_mat.get('blobIndIm')
        # classify full img
        caffe_rep_full = self.net_.evaluate(image)
        if label=='':
            class_guess = np.argmax(caffe_rep_full)
        else: # if the label is provided, we use the GT!
            class_guess = self.net_.get_label_id(label)
            self.topC_ = 0 # hack to force the use of the class_guess
        # Our Obfuscation Search---
        segm_all_list = []
        image_sz = np.shape(image)[0:2]
        for s in range(np.shape(segm_masks)[1]): # for each segm mask
            segm_mask = np.uint16(segm_masks[0,s])
            segm_all = []
            segm_ids = np.unique(segm_mask)
            max_segm_id = np.max(segm_ids)
            confidence = np.zeros(len(segm_ids))
            logging.info('segm_mask {0} / {1} ({2} segments)'.format( \
                         s, np.shape(segm_masks)[1], max_segm_id))
            # Generate the basic segments
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
                    # compute confidence
                    confidence = self.obfuscation_confidence_(image, \
                         segm_mask, id_segment, caffe_rep_full, class_guess, \
                         xmin, xmax, ymin, ymax)
                    # Build the list of segments with positive confidence
                    mask_tmp = np.copy(mask[ymin:ymax,xmin:xmax])
                    bbox = BBox(xmin, ymin, xmax, ymax, max(confidence, 0.0))
                    segm_all.append({'bbox': bbox, 'mask': mask_tmp, \
                                         'id': id_segment})
            # Init the similarity matrix (contains only neighbouring pairs)
            similarity = self.compute_similarity_sets_(segm_all, segm_all, \
                                                       self.alpha_, image_sz)
            similarity = self.zero_diag_values_(similarity)
            # Clustering
            S = list(segm_all)
            segm_mask_supp = np.copy(segm_mask)
            while len(S)!=1:
                # Find highest similarity
                i_max, j_max = np.where(similarity == np.max(similarity))
                i_max = i_max[0]
                j_max = j_max[0]
                # Merge regionsi
                max_segm_id += 1
                segment, segm_mask_supp = self.merge_segments_(S[i_max], \
                                    S[j_max], image, segm_mask_supp, \
                                    max_segm_id, caffe_rep_full, class_guess)
                # Remove Similarity i_max and jmax:
                # Note: order of max and min is to preserve the structure
                similarity = np.delete(similarity, max(i_max, j_max), 0)
                similarity = np.delete(similarity, max(i_max, j_max), 1)
                dummy = S.pop(max(i_max, j_max))
                similarity = np.delete(similarity, min(i_max, j_max), 0)
                similarity = np.delete(similarity, min(i_max, j_max), 1)
                dummy = S.pop(min(i_max, j_max))
                # Add merged segment to S
                S.append(segment)
                assert np.sum(segment['mask'])>0
                # compute similarity of the new segment with the rest
                simtmp = np.zeros((np.shape(similarity)[0] + 1, \
                                    np.shape(similarity)[1] + 1))
                simtmp[0:np.shape(similarity)[0], \
                            0:np.shape(similarity)[1]] = similarity.copy()
                simtmp[:,-1] = self.compute_similarity_sets_([segment], S,\
                                                     self.alpha_, image_sz)
                simtmp[-1,:] = simtmp[:,-1]
                similarity = simtmp.copy()
                similarity = self.zero_diag_values_(similarity)
                # Add merged segment to the output
                segm_all.append(segment)
            segm_all_list.append(segm_all)

        return segm_all_list


    def zero_diag_values_(self, similarity):
        """
        Set to zero the elements on the diagonal (self-similarity) to avoit
        "self-merging". Similarity must be a square matrix.
        """
        for i in range(np.shape(similarity)[0]):
            similarity[i,i] = 0

        return similarity


    def merge_segments_(self, segm_1, segm_2, image, segm_mask_support, \
                        max_segm_id, caffe_rep_full, class_guess):
        """
        Merge two segments in one, and update the segmentation mask replacing
        the old segments with the new one. It also computes the obfuscation
        score for the new segment.
        """
        # Merge bboxes
        xmin = min(segm_1['bbox'].xmin, segm_2['bbox'].xmin)
        ymin = min(segm_1['bbox'].ymin, segm_2['bbox'].ymin)
        xmax = max(segm_1['bbox'].xmax, segm_2['bbox'].xmax)
        ymax = max(segm_1['bbox'].ymax, segm_2['bbox'].ymax)
        # Extract the mask
        id_segment1 = segm_1['id']
        id_segment2 = segm_2['id']
        segm_mask_support[segm_mask_support==id_segment1] = max_segm_id
        segm_mask_support[segm_mask_support==id_segment2] = max_segm_id
        mask = np.copy(segm_mask_support == max_segm_id)
        mask_tmp = np.copy(mask[ymin:ymax,xmin:xmax])
        # perform obfuscation
        conf = self.obfuscation_confidence_(image, segm_mask_support, \
                                    max_segm_id, caffe_rep_full, class_guess,\
                                    xmin, xmax, ymin, ymax)
        # create bbox object
        bbox = BBox(xmin, ymin, xmax, ymax, conf)
        # save output structure
        segment = {'bbox': bbox, 'mask': mask_tmp, 'id': max_segm_id}

        return segment, segm_mask_support


    def obfuscation_confidence_(self, image, segm_mask, id_segment, \
                                caffe_rep_full, class_guess, \
                                xmin, xmax, ymin, ymax):
        """
        Compute the obfuscation score for a given segment with label id_segment
        """
        image_obf = np.copy(image) # copy array
        # If ON, instead of using the segment, we obfuscate the bbox
        # surrouding the segment
        if self.obfuscate_bbox_:
            if np.shape(image.shape)[0]>2: # RGB images
                image_obf[ymin:ymax, xmin:xmax, 0] = \
                                self.net_.get_mean_img()[0]
                image_obf[ymin:ymax, xmin:xmax, 1] = \
                                self.net_.get_mean_img()[1]
                image_obf[ymin:ymax, xmin:xmax, 2] = \
                                self.net_.get_mean_img()[2]
            else: # GRAY images
                image_obf[ymin:ymax, xmin:xmax] = \
                                np.mean(self.net_.get_mean_img())
        else: # If OFF, use segments
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
            confidence = max(caffe_rep_full[class_guess] - \
                            caffe_rep_obf[class_guess], 0.0)
        else:
            idx_sort = np.argsort(caffe_rep_full)[::-1]
            idx_sort = idx_sort[0:self.topC_]
            confidence = max(np.sum(caffe_rep_full[idx_sort] - \
                                    caffe_rep_obf[idx_sort]), 0.0)
        return confidence


    def compute_similarity_sets_(self, segm_set1, segm_set2, alpha, image_sz):
        """
        Compute the similarity matrix between two segment sets, using the
        obfuscation, the fill and the size metrics. They are combined by the
        ndarray alpha of size (3,).
        """
        img_area = np.float(np.prod(image_sz))
        similarity = np.zeros((len(segm_set1), len(segm_set2)))
        for i in range(len(segm_set1)):
            segm_i = segm_set1[i]
            for j in range(len(segm_set2)): # only lower triang
                segm_j = segm_set2[j]
                if True: # self.adjacency_test_(segm_i, segm_j):
                    # Compute new obfuscation score (eulidean dist)
                    s_obf = np.linalg.norm(segm_i['bbox'].confidence-\
                                           segm_j['bbox'].confidence)
                    if self.similarity_:
                        s_obf = 1 - s_obf
                    # Compute size measure
                    s_size = 1 - (np.sum(segm_i['mask']) + \
                                    np.sum(segm_j['mask']))/img_area
                    # Compute fill measure
                    size_BB = (max(segm_i['bbox'].xmax, segm_j['bbox'].xmax) -\
                              min(segm_i['bbox'].xmin, segm_j['bbox'].xmin)) *\
                              (max(segm_i['bbox'].ymax, segm_j['bbox'].ymax) -\
                              min(segm_i['bbox'].ymin, segm_j['bbox'].ymin))
                    s_fill = 1 - (size_BB - np.sum(segm_i['mask']) -
                                    np.sum(segm_j['mask']))/img_area
                    # Merge measures
                    similarity[i,j] = alpha[0]*s_obf + alpha[1]*s_size \
                                        + alpha[2]*s_fill
        return similarity


    def adjacency_test_(segm_1, segm_2):
        """
        Establish if two segments are adjacent.
        """

        raise NotImplementedError()

#----------------- DEPRECATED version
#    def extract(self, image):
#        """
#        Compute segmentation using matlab function, parse the mat files
#        perform segment merging accordingly to the classification score
#        and returns a list of dictionaries
#
#        RETURNS:
#        - a list of dictionaries {'bbox': BBox, 'mask': int nd.array,
#                                  'confidence': int}
#          The bbox is the outer rectable enclosing the segment.
#          The mask contains only 0, 1 values, and it is relative to the bbox.
#          The conficende is max(1 - classification accuracy of obfuscation)
#        """
#       # Print
#        logging.info('Running MATLAB selective search.')
#        # dump the images of the AnnotatedImages to temporary files
#        (fd, img_temp_file) = tempfile.mkstemp(suffix = '.bmp')
#        os.close(fd)
#        if len(np.shape(image))==2: # gray img
#            img = np.zeros((np.shape(image)[0], np.shape(image)[1], 3))
#            img[:,:,0] = image
#            img[:,:,1] = image
#            img[:,:,2] = image
#            image = img
#        img = skimage.io.imsave(img_temp_file, image)
#        # create temporary files for the .mat files
#        (fd, mat_temp_file) = tempfile.mkstemp(suffix = '.mat')
#        os.close(fd)
#        # run the Selective Search Matlab wrapper
#        img_temp_files_cell = '{\'' + img_temp_file + '\'}'
#        mat_temp_files_cell = '{\'' + mat_temp_file + '\'}'
#        matlab_cmd = 'selective_search_obfuscation({0}, {1}, \'{2}\')'\
#                        .format(img_temp_files_cell, mat_temp_files_cell, \
#                                self.ss_version_)
#        command = "matlab -nojvm -nodesktop -r \"try; " + matlab_cmd + \
#                "; catch; exit; end; exit\""
#        logging.info('Executing command ' + command)
#        if os.system(command) != 0:
#            logging.error('Matlab SS script did not exit successfully!')
#            return []
#        # load the .mat file
#        try:
#            segm_mat = scipy.io.loadmat(mat_temp_file)
#        except:
#            logging.error('Exception while loading ' + mat_temp_file)
#        # delete all the temporary files
#        os.remove(img_temp_file)
#        os.remove(mat_temp_file)
#        # Load only first-level segmentation (i.e., Felzenswalb)
#        segm_masks = segm_mat.get('blobIndIm')
#        # classify full img
#        caffe_rep_full = self.net_.evaluate(image)
#        class_guess = np.argmax(caffe_rep_full)
#        # make segm_blobs more "usable" and filter small segments
#        segm_all_list = []
#        for s in range(np.shape(segm_masks)[1]): # for each segm mask
#            segm_mask = segm_masks[0,s]
#            segm_all = []
#            segm_ids = np.unique(segm_mask)
#            max_segm_id = np.max(segm_ids)
#            confidence = np.zeros(len(segm_ids))
#            feature_vec = []
#            logging.info('segm_mask {0} / {1} ({2} segments)'.format( \
#                         s, np.shape(segm_masks)[1], max_segm_id))
#            # compute obfuscation score for each segment
#            #heatmap_tmp = np.zeros((np.shape(image)[0], np.shape(image)[1]))
#            for id_segment in segm_ids: # for each segment of level 0
#                # compute bbox (for filtering)
#                mask = segm_mask==id_segment
#                ys = np.argwhere(np.sum(mask, axis = 1) > 0)
#                xs = np.argwhere(np.sum(mask, axis = 0) > 0)
#                ymin = np.min(ys)
#                ymax = np.max(ys)
#                xmin = np.min(xs)
#                xmax = np.max(xs)
#                if (xmax-xmin >= self.min_sz_segm_) and \
#                        (ymax-ymin >= self.min_sz_segm_): # filter small
#                    image_obf = np.copy(image) # copy array
#                    # obfuscation
#                    if np.shape(image.shape)[0]>2: # RGB images
#                        image_obf[segm_mask==id_segment,0] = \
#                                             self.net_.get_mean_img()[0]
#                        image_obf[segm_mask==id_segment,1] = \
#                                             self.net_.get_mean_img()[1]
#                        image_obf[segm_mask==id_segment,2] = \
#                                             self.net_.get_mean_img()[2]
#                    else: # GRAY images
#                        image_obf[segm_mask==id_segment] = \
#                                       np.mean(self.net_.get_mean_img())
#                    # predict CNN reponse for obfuscation
#                    caffe_rep_obf = self.net_.evaluate(image_obf)
#                    # Given the class of the image, select the confidence
#                    if self.topC_ == 0:
#                        confidence = caffe_rep_full[class_guess] - \
#                                        caffe_rep_obf[class_guess]
#                    else: ### TODOOOOOOOOOOOOOOOOO ###
#                        raise NotImplementedError()
#                    #heatmap_tmp[mask] = confidence
#                    # Build output (bbox and mask)
#                    feature_vec.append([xmin+(xmax-xmin)/2.0, \
#                                        ymin+(ymax-ymin)/2.0, \
#                                        confidence, id_segment])
#                    bbox = BBox(xmin, ymin, xmax, ymax, confidence)
#                    mask_tmp = mask[ymin:ymax,xmin:xmax]
#                    segm_all.append({'bbox': bbox, 'mask': mask_tmp})
#            # Merging segments by confidence [TODOO]
#            logging.info(' - Hierarchical Clustering')
#            X = np.array(feature_vec)[:,0:3] # remove id segm
#            X[:,0] = X[:,0]/np.shape(image)[1] # normalize
#            X[:,1] = X[:,1]/np.shape(image)[0]
#            D = dist.pdist(X[:,0:2], 'euclidean') + self.alpha_ * \
#                dist.pdist(X[:,2].reshape((np.shape(X)[0],1)), 'euclidean')
#            Z = hierarchy.linkage(D, method='average')
#            ZZ = list(segm_all)
#            n = np.shape(Z)[0]
#            segm_mask_support = np.copy(segm_mask)
#            id_segments = np.array(feature_vec)[:,3].tolist()
#            for i in range(n):
#                # Extract the bbox
#                id1, id2, conf, num = Z[i,:]
#                id1 = np.int16(id1)
#                id2 = np.int16(id2)
#                xmin = min(ZZ[id1]['bbox'].xmin, ZZ[id2]['bbox'].xmin)
#                ymin = min(ZZ[id1]['bbox'].ymin, ZZ[id2]['bbox'].ymin)
#                xmax = max(ZZ[id1]['bbox'].xmax, ZZ[id2]['bbox'].xmax)
#                ymax = max(ZZ[id1]['bbox'].ymax, ZZ[id2]['bbox'].ymax)
#                conf = (ZZ[id1]['bbox'].confidence + \
#                            ZZ[id2]['bbox'].confidence)/2.0
#                bbox = BBox(xmin, ymin, xmax, ymax, max(conf, 0.0))
#                # Extract the mask
#                id_segment1 = id_segments[id1]
#                id_segment2 = id_segments[id2]
#                max_segm_id = max_segm_id + 1
#                segm_mask_support[segm_mask_support==id_segment1] = max_segm_id
#                segm_mask_support[segm_mask_support==id_segment2] = max_segm_id
#                mask = np.copy(segm_mask_support == max_segm_id)
#                mask_tmp = mask[ymin:ymax,xmin:xmax]
#                ZZ.append({'bbox': bbox, 'mask': mask_tmp})
#                id_segments.append(max_segm_id)
##                print (id1, id2, len(ZZ)-1)
##                W = hierarchy.dendrogram(Z)
##                pl.show()
##            # visualize
##            import pylab as pl
##            pl.subplot(141)
##            pl.imshow(image)
##            pl.subplot(142)
##            pl.imshow(segm_mask, interpolation='nearest')
##            patches = []
##            confid = []
##            for bbox in ZZ:
##                confid.append(bbox['bbox'].confidence)
##            idx_sort = np.argsort(np.array(confid))[::-1]
##            for i in range(30):
##                bbox = ZZ[idx_sort[i]]
##                patch = pl.Rectangle((bbox['bbox'].xmin, bbox['bbox'].ymin),\
##                           bbox['bbox'].xmax-bbox['bbox'].xmin,\
##                           bbox['bbox'].ymax-bbox['bbox'].ymin, alpha=0.2)
##                pl.gca().add_patch(patch)
##            pl.subplot(143)
##            pl.imshow(segm_mask, interpolation='nearest')
##            labs = []
##            for pos in feature_vec:
##                x,y,c,id = pos
##                pl.scatter(x,y)
##                pl.text(x, y, str(id), fontdict={'size': 18})
##                labs.append(str(id))
##            pl.subplot(144)
##            W = hierarchy.dendrogram(Z, labels = labs)
##            pl.show()
#            segm_all_list.append(ZZ)
#        return segm_all_list
