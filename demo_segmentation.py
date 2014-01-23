import numpy as np
import time
from skimage.data import imread
from xml.dom import minidom
import glob
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *

# Parameters
sigmas = [0.4, 0.5]
mins = [40] 
scales = [300]

def get_label_from_gt(file,gt_path):
    # NOTE: this function is for ImageNet ONLY
    filename = os.path.basename(file) # remove path
    filename = os.path.splitext(filename)[0] # remove file extension
    xmldoc = minidom.parse(gt_path + '/' + filename +'.xml') # parse xml file
    lab_name = xmldoc.getElementsByTagName('name')[0] # get the element 'name' that contains the label name
    lab_name = str(lab_name.firstChild.data) # get the value of the label
    return lab_name

def visualize_partial_results(image, heatmaps, segm_masks):
    # Partial visualization for qualitativa understanding of the results
    import pylab as pl
    n_maps = np.shape(heatmaps)[0]
    for p in range(n_maps):
        #pl.subplot(2,n_maps,2*p+1)
        #pl.imshow(segm_masks[p])
        pl.subplot(1,n_maps,p)
        pl.imshow(heatmaps[p].get_values())
    pl.show()
    
    
## MAIN ##
if __name__ == "__main__":
    conf = Configuration()
    net = NetworkCaffe(conf.ilsvrc2012_caffe_model_spec,\
                       conf.ilsvrc2012_caffe_model,\
                       conf.ilsvrc2012_caffe_wnids_words,\
                       conf.ilsvrc2012_caffe_avg_image)
    # segmentation obj Felzenswalb
    segm = ImgSegmFelzen(scales, sigmas, mins)    
    # heatmap extraction based on segmentation
    heatext = HeatmapExtractorSegm(net, segm, confidence_tech = 'full_obf', area_normalization = False)
    
    # cycle over classes and images in the validation set
    filename_list = glob.glob(conf.ilsvrc2012_val_images+'/*.JPEG')
    file = filename_list[0]
    #for file in filename_list:
    img = imread(file)
    class_label = get_label_from_gt(file,conf.ilsvrc2012_val_box_gt)
    start = time.clock()
    heatmaps = heatext.extract(img, class_label)
    elapsed = (time.clock() - start)
    # Some qualitative analysis
    print 'Elapsed Time: ' + str(elapsed)
    visualize_partial_results(img, heatmaps, segm_masks)       