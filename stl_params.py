import numpy as np

class STLParams:
    # Default parameters
    def __init__(self, min_sz_segm=5, alpha=1/4.0*np.ones((4,)),
                 function_stl = "similarity+cnnfeature",
                 use_fullimg_GT_label=False,topC=5,
                 nms_iou_threshold=0.5,caffe_mode="cpu"):
        # select network: 'CAFFE'
        self.classifier = "CAFFE"
        self.center_only = True
        # Num elements in batch (for decaf/caffe eval)
        self.batch_sz = 1
        self.min_sz_segm = min_sz_segm # keep this low (because we resize!!)
        self.alpha =  alpha
        self.function_stl = function_stl
        self.obfuscate_bbox = True # if false, segments are masked-out (not suggested)
        self.use_fullimg_GT_label = use_fullimg_GT_label # STL_{cl} if true self.topC not used
        self.topC = topC # select topC classes if STL_{u} as in the paper
        self.nms_iou_threshold = nms_iou_threshold # non-maxima suppression param
        self.caffe_mode = caffe_mode# you can run it on "cpu" or "gpu" (suggested)
        # If cnnfeature are used, we can include some padding to the
        # bbox where feature are extracted. value in [0.0, 1.0]
        self.padding = 0.0
        # Select a single parametrization of Feltz alg (i.e., single color space
        # and k)
        self.single_color_space = False
