import numpy as np
from skimage.data import imread
import glob
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *

# Parameters
sigmas = [0.4, 0.6]
mins = [40] 
scales = [100, 300]

def get_label_from_gt(file,gt_path):
    filename = os.path.basename(file)
    # load xml
    print gt_path + '/' + filename
    

## MAIN ##
if __name__ == "__main__":
    conf = Configuration()
    net = NetworkCaffe(conf.ilsvrc2012_caffe_model_spec,\
                       conf.ilsvrc2012_caffe_model,\
                       conf.ilsvrc2012_caffe_wnids_words,\
                       conf.ilsvrc2012_caffe_avg_image)
    # segmentation obj Felzenswalb
    segm = ImgSegmFelzen(sigmas, mins, scales)    
    # heatmap extraction based on segmentation
    heatext = HeatmapExtractorSegm(net, segm, confidence_tech = 'full_obf')
    
    # cycle over classes and images in the validation set
    filename_list = glob.glob(conf.ilsvrc2012_val_images+'/*.JPEG')
    file = filename_list[0]
    #for file in filename_list:
        #img = imread(file)
        #class_label = get_label_from_gt(file,conf.ilsvrc2012_val_box_gt)
        #heatmaps = heatext.extract(img, class_label)
                     