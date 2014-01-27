import numpy as np
import time
from skimage.data import imread
from skimage.transform import resize
from xml.dom import minidom
import glob
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from htmlreport import *

# Parameters
# - segmentation
sigmas = [0.4, 0.5]
mins = [40]
scales = [300]
# - gray box or sliding windows
box_sz = [10, 20, 30]
stride = 5
# resize the biggest dimention of the image to fix_sz
fix_sz = 300

def get_label_from_gt(file,gt_path):
    # NOTE: this function is for ImageNet ONLY
    filename = os.path.basename(file) # remove path
    filename = os.path.splitext(filename)[0] # remove file extension
    xmldoc = minidom.parse(gt_path + '/' + filename +'.xml') # parse xml file
    lab_name = xmldoc.getElementsByTagName('name')[0] # get the label name
    lab_name = str(lab_name.firstChild.data) # get the value of the label
    return lab_name

def visualize_partial_results(image, heatmaps):
    # Partial visualization for qualitativa understanding of the results
    import pylab as pl
    n_maps = np.shape(heatmaps)[0]
    pl.subplot(1,n_maps+1,1)
    pl.imshow(image)
    for p in range(n_maps):
        #pl.subplot(2,n_maps,2*p+1)
        #pl.imshow(segm_masks[p])
        pl.subplot(1,n_maps+1,p+2)
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
    # heatmap extractor
    heatext = HeatmapExtractorSegm(net, segm, confidence_tech = 'full_obf', \
                                   area_normalization = False)
    # Init html object to save results
    htmlres = HtmlReport()

    # cycle over images in the validation set
    filename_list = glob.glob(conf.ilsvrc2012_val_images_dir + '/*.JPEG')
    counter = 1
    #file = filename_list[23425]
    for file in filename_list:
        img = imread(file)
        # rescale max(H,W) ->  fix_sz
        img_sz = np.shape(img)
        max_sz = np.argmax(img_sz)
        prop_sz = fix_sz/float(img_sz[max_sz])
        if max_sz==0:
            img = resize(img, (fix_sz, int(prop_sz*img_sz[1])))
        else:
            img = resize(img, (int(prop_sz*img_sz[0]), fix_sz))
        img = skimage.img_as_ubyte(img)
        class_label = get_label_from_gt(file,conf.ilsvrc2012_val_box_gt)
        start = time.clock()
        heatmaps = heatext.extract(img, class_label)
        elapsed = (time.clock() - start)
        # save results
        htmlres.add_image_embedded(img, proportion = 0.8)
        for p in range(np.shape(heatmaps)[0]):
            htmlres.add_image_embedded(heatmaps[p].get_values(), \
                                       proportion = 0.8)
        # estimate time for each image & print some info
        counter = counter + 1
        print os.path.basename(file) + ', elapsed Time: ' + str(elapsed) + \
              ', process: ' + str(counter/float(np.shape(filename_list)[0])) \
              + '%' + '\n'
        ## Some qualitative analysis
        #visualize_partial_results(img, heatmaps)
        break

    # save html
    htmlres.save('results.html')
