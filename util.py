import numpy as np
import skimage
import skimage.transform

def resize_image_max_size(img, fix_sz):
    """
    Return a resized version of the image, where the longest edge has 
    length 'fix_sz' pixels. The resizing mantains the proportion.
    """
    img = np.copy(img)
    great_size = np.max(img.shape)
    if great_size > fix_sz:
        proportion = fix_sz / float(great_size)
        width = int(img.shape[1] * float(proportion))
        height = int(img.shape[0] * float(proportion))    
        img = skimage.transform.resize(img, (height, width))
    return img

def crop_image_center(img):
    """
    Returns the crop of the image, made by taking the central region.
    """
    img = np.copy(img)
    if img.shape[0] < img.shape[1]:
        offset = (img.shape[1] - img.shape[0]) / 2
        img = img[0:img.shape[0], offset:(offset+img.shape[0])]
    else:
        offset = (img.shape[0] - img.shape[1]) / 2
        img = img[offset:(offset+img.shape[1]), 0:img.shape[1]]
    return img

