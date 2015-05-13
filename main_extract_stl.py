import numpy as np
import os
import os.path
import skimage.io as skio
import skimage
import glob
#from vlg.util.parfun import *

from util import *
from configuration import *
from bbox import *
from imgsegmentation import *
from network import *
from self_taught_loc import *

class Params:
    # Default parameters
    def __init__(self):
        # If cnnfeature are used, we can include some padding to the
        # bbox where feature are extracted. value in [0.0, 1.0]
        self.padding = 0.0
        # Select a single parametrization of Feltz alg (i.e., single color space
        # and k)
        self.single_color_space = False
        # select network: 'CAFFE'
        self.classifier = 'CAFFE'
        self.center_only = True
        # Num elements in batch (for decaf/caffe eval)
        self.batch_sz = 1


if __name__ == "__main__":
    # --- Config and Parameters --- #
    # load configurations and parameters
    # load_Girshick_caffe_model = True
    conf = Configuration()
    params = Params()
    # experiment name.
    params.exp_name = 'segtrackv1'  # full path is /home/ironfs/scratch/karim/segtrackv1
    params.sub_exp  = 'penguin'        # sub experiment or 1 problem in dataset     
    # default Configuration, image and label files
    params.conf = conf
    # select top C classes used to generate the predicted bboxes
    params.topC = 5     # if 0, take the max across classes
    # obfuscation search params
    params.min_sz_segm = 5 # keep this low (because we resize!!)
    params.alpha =  1/4.0*np.ones((4,))
    params.function_stl = 'similarity+cnnfeature'
    params.obfuscate_bbox = True
    params.use_fullimg_GT_label = False # if true params.topC is not used!
    # Non-maxima suppression
    params.nms_execution = True
    params.nms_iou_threshold = 0.5
    # Save results to disk
    params.dump_results = True
    # input/output directory
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name + '/' + params.sub_exp
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # --- Run experiment --- #
    # Instanciate network
    netParams = NetworkCaffeParams(conf.ilsvrc2012_caffe_model_spec, \
                           conf.ilsvrc2012_caffe_model, \
                           conf.ilsvrc2012_caffe_wnids_words, \
                           conf.ilsvrc2012_caffe_avg_image, \
                           center_only = params.center_only,\
                           wnid_subset = [])
    net = Network.create_network(netParams)
    # feat = net.evaluate(img, layer_name = 'fc7')
    # Instanciate the obj segmentation algorithm
    img_segmenter = ImgSegmMatWraper()
    # Instantiate STL object
    stl_grayout = SelfTaughtLoc_Grayout(net, img_segmenter, \
                            params.min_sz_segm, topC = params.topC,\
                            alpha = params.alpha, \
                            obfuscate_bbox = params.obfuscate_bbox, \
                            function_stl = params.function_stl,\
                            padding = params.padding,\
                            single_color_space = params.single_color_space)
    # Load img and resize to the net input   
    currentPath = 'test_data/' + params.exp_name + '/' + params.sub_exp + '/'
    segTestImgs = glob.glob(currentPath + '*bmp')
    for full_image_name in segTestImgs:
        print "=================== Processing image: " + full_image_name + " ================================"
        image_name = os.path.split(full_image_name)[-1]
        img = skio.imread(full_image_name)
        img_height, img_width = np.shape(img)[0:2]
        gt_labels = [] # must be provided if use the supervised method
        image_resz = skimage.transform.resize(img,\
                        (net.get_input_dim(), net.get_input_dim()))
        image_resz = skimage.img_as_ubyte(image_resz)
        img_height_resz, img_width_resz = np.shape(image_resz)[0:2]
        # Extract STL
        segment_lists = {}
        if not params.use_fullimg_GT_label:
            this_label = 'none'
            segment_lists[this_label] = stl_grayout.extract_greedy(image_resz)
        else:
            for GT_label in  gt_labels:
                segment_lists[GT_label] = \
                            stl_grayout.extract_greedy(image_resz, label=GT_label)
        pred_objects = {}
        bboxes = []
        for this_label in segment_lists.keys():
            if not params.use_fullimg_GT_label:
                assert this_label == 'none'
            # Convert the segmentation lists to BBoxes
            pred_bboxes_unnorm = segments_to_bboxes(segment_lists[this_label])
            # Normalize the bboxes
            pred_bboxes = []
            for j in range(np.shape(pred_bboxes_unnorm)[0]):
                pred_bboxes_unnorm[j].normalize_to_outer_box(BBox(0, 0, \
                                                    img_width_resz-1, img_height_resz-1))
                pred_bboxes.append(pred_bboxes_unnorm[j])
            # execute NMS, if requested
            if params.nms_execution:
                pred_bboxes = BBox.non_maxima_suppression( \
                                        pred_bboxes, params.nms_iou_threshold)
            # results
            pred_objects[this_label] = pred_bboxes
            # accumulate class-agnostic results to store them into disk
            bboxes.extend(pred_bboxes)
        # dump bboxes to txt files
        if params.dump_results:
            # rescale to img original size
            for bb in bboxes:
                bb.rescale_to_outer_box(img_width, img_height)
                bb.convert_coordinates_to_integers()
            outfilename = params.output_dir + '/' + image_name + '.txt'
            success = BBox.dump_bboxes_to_file(bboxes, outfilename)
            assert success == 1
            logging.info('Output files in {0}'.format(params.output_dir))