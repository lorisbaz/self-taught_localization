import numpy as np
from skimage import io
import skimage.transform
import os
try:
    import decaf.scripts.imagenet
    from decaf.scripts.imagenet import DecafNet
    import decaf.util.transform
except:
    print "Warning: Decaf not loaded. \n"
try:
    import caffe.imagenet
except:
    print "Warning: Caffe not loaded. \n"

import util


class Network:
    """Network class"""

    def __init__(self):
        raise NotImplementedError()

    def evaluate(self, img, layer_name = 'softmax'):
        """Evaluate the network given a an image, and export the layer
        named 'layer_name'.
        The 'layer_name' can be:
        - 'softmax'
        - 'fc7_relu' (Note: Decaf only)
        - 'fc7' (Note: Decaf only)
        - 'fc6_relu' (Note: Decaf only)
        - 'fc6' (Note: Decaf only)
        - 'pool5' (Note: Decaf only)

        image: ndarray.uint8
        layer_name: string
        """
        raise NotImplementedError()

    def get_mean_img(self):
        """
        Returns the unique label id \in {0, ..., num_labels-1}
        """
        raise NotImplementedError()

    def get_input_dim(self):
        """
        Returns the input size of the network (scalar value)
        """

    def get_label_id(self, label):
        """
        Returns the unique label id \in {0, ..., num_labels-1}
        """
        raise NotImplementedError()

    def get_label_desc(self, label):
        """
        Returns a textual description of the label
        """
        raise NotImplementedError()

    def get_labels(self):
        """
        Returns the list of labels (in order of id) that the network can use
        """
        raise NotImplementedError()

#=============================================================================

class NetworkDecaf(Network):
    """
    Implementation for the Decaf library.
    """
    def __init__(self, model_spec_filename, model_filename,\
                 wnid_words_filename, center_only = False):
        # load Decaf model
        self.net_ = DecafNet(model_filename, model_spec_filename)
        self.center_only_ = center_only
        # build a dictionary label --> description
        self.dict_label_desc_ = {}
        dict_desc_label = {}
        fd = open(wnid_words_filename)
        for line in fd:
            temp = line.strip().split('\t')
            wnid = temp[1].strip()
            self.dict_label_desc_[wnid] = temp[2].strip()
            dict_desc_label[temp[2].split(',')[0]] = wnid
        fd.close()
        # build a dictionary label --> label_id
        self.dict_label_id_ = {}
        self.labels_ = []
        for i, desc in enumerate(self.net_.label_names):
            self.dict_label_id_[dict_desc_label[desc]] = i
            self.labels_.append(dict_desc_label[desc])
        # Load the mean vector from file
        # mean of 3 channels
        self.net_.mean_img =np.mean(np.mean(self.net_._data_mean,axis=1),axis=0)
        # it is in BGR convert in RGB
        #self.net_.mean_img = self.net_.mean_img[::-1]

    def get_mean_img(self):
        return self.net_.mean_img

    def get_input_dim(self):
        return decaf.scripts.imagenet.INPUT_DIM

    def get_label_id(self, label):
        return self.dict_label_id_[label]

    def get_label_desc(self, label):
        return self.dict_label_desc_[label]

    def get_labels(self):
        return self.labels_

    def evaluate(self, img, layer_name = 'softmax'):
        # for now only center_only is supported
        assert self.center_only_ == True
        # first, extract the 227x227 center
        dim = decaf.scripts.imagenet.INPUT_DIM
        image = util.crop_image_center(decaf.util.transform.as_rgb(img))
        image = skimage.transform.resize(image, (dim, dim))
        # convert to [0,255] float32
        image = image.astype(np.float32) * 255.
        assert np.max(image) <= 255
        # Flip the image if necessary, maintaining the c_contiguous order
        if decaf.scripts.imagenet._JEFFNET_FLIP:
            image = image[::-1, :].copy()
        # subtract the mean, cropping the 256x256 mean image
        xoff = (self.net_._data_mean.shape[1] - dim)/2
        yoff = (self.net_._data_mean.shape[0] - dim)/2
        image -= self.net_._data_mean[yoff+yoff+dim, xoff:xoff+dim]
        # make sure the data in contiguous in memory
        images = np.ascontiguousarray(image[np.newaxis], dtype=np.float32)
        # classify
        predictions = self.net_.classify_direct(images)
        scores = predictions.mean(0)        
        # look at the particular layer
        if layer_name == 'softmax':
            return scores
        elif layer_name == 'fc7_relu':
            layer_name = 'fc7_neuron_cudanet_out'
        elif layer_name == 'fc7':
            layer_name = 'fc7_cudanet_out'
        elif layer_name == 'fc6_relu':
            layer_name = 'fc6_neuron_cudanet_out'
        elif layer_name == 'fc6':
            layer_name = 'fc6_cudanet_out'
        elif layer_name == 'pool5':
            layer_name = 'pool5_cudanet_out'
        else:
            raise ValueError('layer_name not supported')
        return self.net_.feature(layer_name)

#=============================================================================

class NetworkCaffe(Network):
    """
    Implementation for the Caffe library.
    """
    def __init__(self, model_spec_filename, model_filename,\
                 wnid_words_filename, mean_img_filename, \
                 caffe_mode = 'cpu', center_only = False):
        # for now, we support only the single full-image evaluation
        assert center_only == True
        # load Caffe model
        self.net_ = caffe.imagenet.ImageNetClassifier( \
                            model_spec_filename, model_filename, \
                            center_only)
        self.net_.caffenet.set_phase_test()
        if caffe_mode == 'cpu':
            self.net_.caffenet.set_mode_cpu()
        elif caffe_mode == 'gpu':
            self.net_.caffenet.set_mode_gpu()
        else:
            raise ValueError('caffe_mode not recognized')
        # build a dictionary label --> description
        # and a dictionary label --> label_id
        self.dict_label_desc_ = {}
        self.dict_label_id_ = {}
        self.labels_ = []
        fd = open(wnid_words_filename)
        line_number = 0
        for line in fd:
            wnid = line[0:9].strip()
            words = line[10:].strip()
            self.dict_label_desc_[wnid] = words
            self.dict_label_id_[wnid] = line_number
            self.labels_.append(wnid)
            line_number += 1
        fd.close()
        # Load the mean vector from file
        self.net_.mean_img = np.load(\
                   os.path.join(os.path.dirname(__file__), mean_img_filename))
        # mean of 3 channels
        self.net_.mean_img = np.mean(np.mean(self.net_.mean_img,axis=1),axis=0) 
        # it is in BGR convert in RGB
        self.net_.mean_img = self.net_.mean_img[::-1] 

    def get_mean_img(self):
        return self.net_.mean_img

    def get_input_dim(self):
        return caffe.imagenet.CROPPED_DIM

    def get_label_id(self, label):
        return self.dict_label_id_[label]

    def get_label_desc(self, label):
        return self.dict_label_desc_[label]

    def get_labels(self):
        return self.labels_

    def evaluate(self, img, layer_name = 'softmax'):
        # if the image in in grayscale, we make it to 3-channels
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # first, extract the 227x227 center, and convert it to BGR
        dim = self.get_input_dim()
        image = util.crop_image_center(decaf.util.transform.as_rgb(img))
        image_reshape = skimage.transform.resize(image, (dim, dim))
        image_reshape = (image_reshape * 255)[:, :, ::-1]
        # subtract the mean, cropping the 256x256 mean image
        xoff = (caffe.imagenet.IMAGENET_MEAN.shape[1] - dim)/2
        yoff = (caffe.imagenet.IMAGENET_MEAN.shape[0] - dim)/2
        image_reshape -= caffe.imagenet\
                            .IMAGENET_MEAN[yoff+yoff+dim, xoff:xoff+dim]
        # oversample code
        image = image_reshape.swapaxes(1, 2).swapaxes(0, 1)
        input_blob = [np.ascontiguousarray(image[np.newaxis], dtype=np.float32)]
        # forward pass to the network
        num = 1
        num_output=1000
        output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
        self.net_.caffenet.Forward(input_blob, output_blobs)
        scores = output_blobs[0].mean(0).flatten()
        assert layer_name == 'softmax', 'layer_name not supported'
        return scores
