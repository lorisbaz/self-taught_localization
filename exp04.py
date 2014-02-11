import numpy as np
import os
import os.path
import skimage.io
from vlg.util.parfun import *
import logging

from heatmap import *
from network import *
from configuration import *
from imgsegmentation import *
from heatextractor import *
from htmlreport import *
from util import *

class Params:
    def __init__(self):
        pass

def get_filenames(params):
    """
    Return a list of (wnid, filename) for all the files
    with label_id \in {1, ..., num_classe}
    """
    conf = params.conf
    wnids = []
    fd = open(conf.ilsvrc2012_classid_wnid_words)
    for line in fd:
        temp = line.strip().split('\t')
        wnids.append(temp[1].strip())
    fd.close()
    images = []
    labels_id = []
    fd = open(params.images_file)
    for line in fd:
        images.append(conf.ilsvrc2012_root_images_dir + '/' + line.strip())
    fd.close()
    fd = open(params.labels_file)
    for line in fd:
        labels_id.append(int(line.strip()))
    fd.close()
    assert len(images) == len(labels_id)
    # return only the labels 1..num_classes
    out = []
    for i in range(len(images)):
        if labels_id[i] > params.num_classes:
            continue
        out.append( (wnids[labels_id[i]-1], images[i]) )
    return out


def pipeline(images, output_html, params):
    """
    Run the pipeline for this experiment. images is a list of
    (wnid, image_filename)
    """
    # Instantiate some objects
    conf = params.conf
    net = NetworkDecaf(conf.ilsvrc2012_decaf_model_spec, \
                       conf.ilsvrc2012_decaf_model, \
                       conf.ilsvrc2012_classid_wnid_words, \
                       center_only = True)
    segmenter = ImgSegmFromMatFiles(conf.ilsvrc2012_segm_results_dir, \
    			    conf.ilsvrc2012_root_images_dir, \
    			    params.fix_sz)
    heatext = HeatmapExtractorSegm(net, segmenter, \
                    confidence_tech = params.heatextractor_confidence_tech, \
                   area_normalization = params.heatextractor_area_normalization)
    htmlres = HtmlReport()
    # loop over the images
    for image in images:
        image_wnid, image_file = image
        logging.info('Elaborating ' + os.path.basename(image_file))
        img = skimage.io.imread(image_file)
        # rescale the image (if necessary), and crop it to the central region
        img = resize_image_max_size(img, params.fix_sz)
        img = skimage.img_as_ubyte(img)
        img = crop_image_center(img)
        # sync segmentation loader	
        segmenter.set_image_name(image_file)
        # add the image to the html
        logging.info('Image size: ({0}, {1})'.format(img.shape[0], img.shape[1]))
        desc = '{0}\n{1}'.format(image_wnid, os.path.basename(image_file))
        htmlres.add_image_embedded(img, max_size = params.html_max_img_size, \
    	                           text = desc)
        # extract the segmentation masks
        if params.visualize_segmentation_masks:
            seg_masks = segmenter.extract(img)
            for idx, seg in enumerate(seg_masks):
                num_segments = np.max(seg)+1
                desc = 'seg {0} {1} (num_segs: {2})'\
                    .format(idx, str(np.shape(seg_masks)[0]), num_segments)
                seg_img = np.float32(seg) / float(num_segments)
                seg_img = skimage.img_as_ubyte(seg_img)
                htmlres.add_image_embedded(seg_img, \
                        max_size = params.html_max_img_size, \
                        text = desc)
        # extract the heatmaps
        if params.visualize_heatmaps:
            heatmaps = heatext.extract(img, image_wnid)
            for idx, heatmap in enumerate(heatmaps):
                desc = 'heatmap {0}'.format(idx)
                htmlres.add_image_embedded( \
                heatmap.export_to_image(factor = params.visual_factor), \
                            max_size = params.html_max_img_size, \
                            text = desc)
            desc = 'AVG seg'
            heatmap_avg = Heatmap.sum_heatmaps(heatmaps)
            heatmap_avg.normalize_counts()
            htmlres.add_image_embedded( \
   	        heatmap_avg.export_to_image(factor = params.visual_factor), \
                           max_size = params.html_max_img_size, \
                           text = desc)
        htmlres.add_newline()
        # save html and exit
        htmlres.save(output_html)
    return 0


def run_exp(params):
    # create output directory
    if os.path.exists(params.output_dir) == False:
        os.makedirs(params.output_dir)
    # load the filenames of the first 10 classes of
    # ILSVRC2012-validation, and divide the images by class
    images = get_filenames(params)
    images = sorted(images, key=lambda x: x[0])
    images_by_class = [[] for i in range(params.num_classes)]
    assert len(images_by_class) == params.num_classes
    current_wnid = images[0][0]
    current_idx = 0
    for image in images:
	if image[0] != current_wnid:
	    current_idx += 1
	    current_wnid = image[0]
	images_by_class[current_idx].append(image)
    assert current_idx == params.num_classes-1
    # run the pipeline
    parfun = None
    if params.run_on_anthill:
    	parfun = ParFunAnthill(pipeline)
    else:
        parfun = ParFunDummy(pipeline)
    for i, images in enumerate(images_by_class):
        output_html = '%s/%02d.html' % (params.output_dir, i)
        parfun.add_task(images[0:min(len(images), \
                        params.num_images_per_class)], \
                        output_html, \
			            params)
    out = parfun.run()
    for i, val in enumerate(out):
        if val != 0:
            logging.info('Task {0} didn''t exit properly'.format(i))
    logging.info('End of the script')
