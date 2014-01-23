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
    class_list = glob.glob(conf.ilsvrc2012_val_images+'/*')
    for c in class_list:
        image_list = glob.glob(class_list)
        for i in image_list:
            img = imread(i)
            #heatmaps = heatext.extract(img, i)
                     