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
