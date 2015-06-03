# 3rd-part libs
import numpy as np
import skimage.io as skio
import skimage
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# my libs
from stl_params import *
from util import *
from configuration import *
from bbox import *
from imgsegmentation import *
from network import *
from self_taught_loc import *

if __name__ == "__main__":
    # 0) Config and Parameters
    root = "/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012/caffe_model_141118"
    conf = Configuration(root=root)
    params = STLParams(caffe_mode="gpu") # look insize stl_params to set your own params
    # input image
    full_image_name = "./test_data/ILSVRC2012_val_00000001_n01751748.JPEG"
    gt_labels = [] # must be provided if use STL_{cl}

    # 1) Init network and STL object
    netParams = NetworkCaffe1114Params(conf.caffe_model_spec, \
                           conf.caffe_model, \
                           conf.caffe_wnids_words, \
                           conf.caffe_avg_image, \
                           caffe_mode = params.caffe_mode, \
                           center_only = params.center_only,\
                           wnid_subset = [])
    net = Network.create_network(netParams)
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

    # 2) Load img and resize to the net input
    logging.info("=> Processing image: {0}".format(full_image_name))
    img = skio.imread(full_image_name)
    img_height, img_width = np.shape(img)[0:2] # original sizes
    image_resz = skimage.transform.resize(img,\
                    (net.get_input_dim(), net.get_input_dim())) # resize
    image_resz = skimage.img_as_ubyte(image_resz)
    img_height_resz, img_width_resz = np.shape(image_resz)[0:2] # resized sizes

    # 3) Extract STL_{u} or STL_{cl}
    segment_lists = {}
    if not params.use_fullimg_GT_label: # STL_{u} unsupervised
        this_label = "none"
        segment_lists[this_label] = stl_grayout.extract_greedy(image_resz)
    else: # STL_{cl} supervised
        for GT_label in gt_labels:
            segment_lists[GT_label] = \
                            stl_grayout.extract_greedy(image_resz, label=GT_label)
    # 4) Post-process results
    pred_objects = {}
    bboxes = []
    for this_label in segment_lists.keys():
        if not params.use_fullimg_GT_label:
            assert this_label == "none"
        # Convert the segmentation lists to BBoxes
        pred_bboxes_unnorm = segments_to_bboxes(segment_lists[this_label])
        # Normalize the bboxes
        pred_bboxes = []
        for j in range(np.shape(pred_bboxes_unnorm)[0]):
            pred_bboxes_unnorm[j].normalize_to_outer_box(BBox(0, 0, \
                                                    img_width_resz-1, img_height_resz-1))
            pred_bboxes.append(pred_bboxes_unnorm[j])
        # execute NMS
        pred_bboxes = BBox.non_maxima_suppression( \
                                        pred_bboxes, params.nms_iou_threshold)
        # results
        pred_objects[this_label] = pred_bboxes
        # accumulate class-agnostic results to store them into disk
        bboxes.extend(pred_bboxes)
    # rescale to img original sizes
    for bb in bboxes:
        bb.rescale_to_outer_box(img_width, img_height)
        bb.convert_coordinates_to_integers()

    # 5) visualize results after rescaling
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for bb in bboxes:
        rect = mpatches.Rectangle((bb.xmin, bb.ymin),
                                  bb.xmax-bb.xmin, bb.ymax-bb.ymin,
                                    fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
    plt.show()
