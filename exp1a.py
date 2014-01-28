import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *

from heatmap import *
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from htmlreport import *

# experiment name
exp_name = 'exp1a'
# number of classes to elaborate
num_classes = 2
# number of images per class to elaborate
num_images_per_class = 3
# default Configuration, image and label files
conf = Configuration()
images_file = conf.ilsvrc2012_val_images
labels_file = conf.ilsvrc2012_val_labels
# Felzenswalb segmentation params: (scale, sigma, min)
seg_params = [(200, 0.1, 400), \
              (200, 0.5, 400), \
              (200, 1.0, 400), \
              (400, 0.1, 400), \
              (400, 0.5, 400), \
              (400, 1.0, 400), \
              (800, 0.1, 400), \
              (800, 0.5, 400), \
              (800, 1.0, 400)]
# we first resize each image to this size, if bigger 
fix_sz = 300
# the maximum size of an image in the html files
html_max_img_size = 300
# method for calculating the confidence
heatextractor_confidence_tech = 'full_obf'
# normalize the confidence by area?
heatextractor_area_normalization = False
# output directory
output_dir = conf.experiments_output_directory + '/' + exp_name
# parallelize the script on Anthill?
run_on_anthill = False
# segmentation masks?
visualize_segmentation_masks = True
# visualize heatmaps?
visualize_heatmaps = True


def pipeline(images, output_html):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename)
    """
    # Instantiate some objects
    net = NetworkCaffe(conf.ilsvrc2012_caffe_model_spec, \
                       conf.ilsvrc2012_caffe_model, \
                       conf.ilsvrc2012_caffe_wnids_words, \
                       conf.ilsvrc2012_caffe_avg_image)
    segmenter = ImgSegmFelzen(params = seg_params)
    heatext = HeatmapExtractorSegm( \
       net, segmenter, confidence_tech = heatextractor_confidence_tech, \
       area_normalization = heatextractor_area_normalization)
    htmlres = HtmlReport()
    # loop over the images
    for image in images:
        image_wnid, image_file = image
        print 'Elaborating ' + os.path.basename(image_file)
        img = skimage.io.imread(image_file)
        # rescale the image (if necessary)
        great_size = np.max(img.shape)
        if great_size > fix_sz:
            proportion = fix_sz / float(great_size)
            width = int(img.shape[1] * float(proportion))
            height = int(img.shape[0] * float(proportion))    
            img = skimage.transform.resize(img, (height, width))
        img = skimage.img_as_ubyte(img)
        print 'Image size: ({0}, {1})'.format(img.shape[0], img.shape[1])
        desc = '{0}\n{1}'.format(image_wnid, os.path.basename(image_file))
        htmlres.add_image_embedded(img, max_size = html_max_img_size, \
                                    text = desc)
        # extract the segmentation masks
        if visualize_segmentation_masks:
            seg_masks = segmenter.extract(img)
            for idx, seg in enumerate(seg_masks):
                desc = 'seg {0} {1}'.format(idx, str(seg_params[idx]))
                num_segments = np.max(seg)+1
                seg_img = np.float32(seg) / float(num_segments)
                seg_img = skimage.img_as_ubyte(seg_img)
                htmlres.add_image_embedded(seg_img, max_size = html_max_img_size, \
                                           text = desc)
        # extract the heatmaps
        if visualize_heatmaps:
            heatmaps = heatext.extract(img, image_wnid)
            for idx, heatmap in enumerate(heatmaps):
                desc = 'heatmap {0}'.format(idx) 
                htmlres.add_image_embedded(heatmap.export_to_image(), \
                                           max_size = html_max_img_size, \
                                           text = desc)
        htmlres.add_newline()
    # save html and exit
    htmlres.save(output_html)
    return 0
    

def get_filenames():
    """
    Return a list of (wnid, filename) for all the files 
    with label_id \in {1, ..., num_classe}
    """
    wnids = []
    fd = open(conf.ilsvrc2012_classid_wnid_words)
    for line in fd:
        temp = line.strip().split('\t')
        wnids.append(temp[1].strip())
    fd.close()
    images = []
    labels_id = []
    fd = open(images_file)
    for line in fd:
        images.append(conf.ilsvrc2012_root_images_dir + '/' + line.strip())
    fd.close()
    fd = open(labels_file)
    for line in fd:
        labels_id.append(int(line.strip()))
    fd.close()
    assert len(images) == len(labels_id)
    # return only the labels 1..num_classes
    out = []
    for i in range(len(images)):
        if labels_id[i] > num_classes:
            continue
        out.append( (wnids[labels_id[i]-1], images[i]) )
    return out


if __name__ == "__main__":
    # create output directory
    if os.path.exists(output_dir) == False:
        os.makedirs(output_dir)
    # load the filenames of the first 10 classes of 
    # ILSVRC2012-validation, and divide the images by class
    images = get_filenames()
    images = sorted(images, key=lambda x: x[0])
    images_by_class = [[] for i in range(num_classes)]
    assert len(images_by_class) == num_classes
    current_wnid = images[0][0]
    current_idx = 0
    for image in images:
        if image[0] != current_wnid:
            current_idx += 1
            current_wnid = image[0]
        images_by_class[current_idx].append(image)
    assert current_idx == num_classes-1
    # run the pipeline
    parfun = None
    if run_on_anthill:
        parfun = ParFunAnthill(pipeline)
    else:
        parfun = ParFunDummy(pipeline)
    for i, images in enumerate(images_by_class):
        output_html = '%s/%02d.html' % (output_dir, i)
        parfun.add_task(images[0:min(len(images), num_images_per_class)], \
                        output_html)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            print 'Task {0} didn''t exit properly'.format(i)
    print 'End of the script'
