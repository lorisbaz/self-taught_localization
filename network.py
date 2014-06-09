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
import matplotlib.pyplot as plt

import util

class NetworkParams:
    """ Parameters for the Network class """
    def __init__(self):
        raise NotImplementedError()

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

    @staticmethod
    def create_network(params):
        """
        Factory for the Network, taking a NetoworkParams as input.
        """
        assert isinstance(params, NetworkParams)
        if isinstance(params, NetworkFakeParams):
            return NetworkFake(params)
        elif isinstance(params, NetworkDecafParams):
            return NetworkDecaf(params)
        elif isinstance(params, NetworkCaffeParams):
            return NetworkCaffe(params)
        else:
            raise ValueError('NetworkParams instance not recognized')

#=============================================================================

class NetworkFakeParams(NetworkParams):
    def __init__(self):
        pass

class NetworkFake(Network):
    """Fake Network class for debugging purposes"""

    def __init__(self, params):
        pass

    def evaluate(self, img, layer_name = 'softmax'):
        if img == 'image0':
            return np.zeros(shape=(1,1000), dtype=float)
        elif img == 'image1':
            return np.ones(shape=(1,1000), dtype=float)
        else:
            return np.multiply(np.ones(shape=(1,1000), dtype=float), 99.0)

    def get_mean_img(self):
        raise NotImplementedError()

    def get_input_dim(self):
        return 227

    def get_label_id(self, label):
        if label == 'label0':
            return 0
        elif label == 'label1':
            return 1
        elif label == 'label2':
            return 2
        else:
            NotImplementedError()

    def get_label_desc(self, label):
        return 'DESCRIPTION' + label

    def get_labels(self):
        return ['label0', 'label1', 'label2']

#=============================================================================

class NetworkDecafParams(NetworkParams):
    def __init__(self, model_spec_filename, model_filename,\
                 wnid_words_filename, center_only = False, wnid_subset = []):
        self.model_spec_filename = model_spec_filename
        self.model_filename = model_filename
        self.wnid_words_filename = wnid_words_filename
        self.center_only = center_only
        self.wnid_subset =  wnid_subset

class NetworkDecaf(Network):
    """
    Implementation for the Decaf library.
    """
    def __init__(self, model_spec_filename, model_filename=None,\
                 wnid_words_filename=None, center_only=False, wnid_subset = []):
        """
        *** PRIVATE CONSTRUCTOR ***
        """
        # the following is just an hack to allow retro-compatibility
        # with existing code
        if isinstance(model_spec_filename, NetworkDecafParams):
            params = model_spec_filename
            model_spec_filename = params.model_spec_filename
            model_filename = params.model_filename
            wnid_words_filename = params.wnid_words_filename
            center_only = params.center_only
            wnid_subset = params.wnid_subset
            if wnid_subset!=[]:
                print 'Warning: subset of labels not supported yet'
        else:
            assert isinstance(model_spec_filename, str)
            assert model_filename != None
            assert wnid_words_filename != None
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

class NetworkCaffeParams(NetworkParams):
    def __init__(self, model_spec_filename, model_filename,\
                 wnid_words_filename, mean_img_filename,\
                 caffe_mode = 'cpu', center_only = False, wnid_subset = []):
        self.model_spec_filename = model_spec_filename
        self.model_filename = model_filename
        self.wnid_words_filename = wnid_words_filename
        self.mean_img_filename = mean_img_filename
        self.caffe_mode = caffe_mode
        self.center_only = center_only
        self.wnid_subset = wnid_subset

class NetworkCaffe(Network):
    """
    Implementation for the Caffe library.
    """
    def __init__(self, model_spec_filename, model_filename=None,\
                 wnid_words_filename=None, mean_img_filename=None, \
                 caffe_mode='cpu', center_only=False, wnid_subset = []):
        """
        *** PRIVATE CONSTRUCTOR ***
        """
        # the following is just an hack to allow retro-compatibility
        # with existing code
        if isinstance(model_spec_filename, NetworkCaffeParams):
            params = model_spec_filename
            model_spec_filename = params.model_spec_filename
            model_filename = params.model_filename
            wnid_words_filename = params.wnid_words_filename
            mean_img_filename = params.mean_img_filename
            caffe_mode = params.caffe_mode
            center_only = params.center_only
            try:
                wnid_subset = params.wnid_subset
            except: # this was done for the detection challenge 2013
                wnid_subset = []
        else:
            assert isinstance(model_spec_filename, str)
            assert model_filename != None
            assert wnid_words_filename != None
            assert mean_img_filename != None
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
        # extract the list of layers
        try:
            self.layer_list = [k for k, dummy in \
                                    self.net_.caffenet.blobs.items()]
        except:
            print 'Warning: Old version of Caffe, please install a more ' \
                                'recent version (23 April 2014) or only ' \
                                'softmax output layer is supported.'
        # subset of ids
        self.wnid_subset = wnid_subset

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
        """
        Evaluates an image with caffe and extracts features at the layer_name.
        layer_name can assume different values dependengly on the network
        architecture that you are using.
        Most common names are:
        - 'prob' or 'softmax' (default): for the last layer representation
            usually used for classitication
        - 'fc<N>', <N> is the level number: the fully connected layers
        - 'conv<N>': the convolutional layers
        - 'pool<N>': the pooling layers
        - 'norm<N>': the fully connected layers
        """
        # if the image in in grayscale, we make it to 3-channels
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # first, extract the 227x227 center, and convert it to BGR
        dim = self.get_input_dim()
        image = util.crop_image_center(img)
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
        try:
            last_layer = self.net_.caffenet.blobs.items()[-1]
            num_output = len(last_layer[1].data.flatten())
        except: # it means you have the old version of caffe
            num_output = 1000
        output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
        self.net_.caffenet.Forward(input_blob, output_blobs)
        #assert layer_name == 'softmax', 'layer_name not supported'
        # NOTE: decaf and caffe have different name conventions (keep softmax
        #       for back-compatibility)
        if layer_name == 'softmax':
            layer_name = 'prob'
        try:
            net_representation = {k: output for k, output in \
                                    self.net_.caffenet.blobs.items()}
            if net_representation.has_key(layer_name):
                scores = net_representation[layer_name].data[0]
                assert np.shape(net_representation[layer_name].data)[0] == 1
                # Done for back-compatibility (remove single dimentions)
                if np.shape(scores)[1]==1 and np.shape(scores)[2]==1:
                    scores = scores.flatten()
            else:
                raise ValueError('layer_name not supported')
        except:
            scores = output_blobs[0].mean(0).flatten()
            print 'Warning: Old version of Caffe, please install a more ' \
                                'recent version (23 April 2014). The softmax' \
                                'layer is now output of this function.'
        # if a subset is provided, we zero out the entry not used
        if self.wnid_subset != [] and layer_name == 'prob':
            for wnid in self.labels_:
                if wnid not in self.wnid_subset:
                    scores[self.get_label_id(wnid)] = 0.0
        return scores

    def extract_all(self, img):
        """
        Evaluates an image with caffe and extracts features for each layer.
        """
        # if the image in in grayscale, we make it to 3-channels
        if img.ndim == 2:
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        # first, extract the 227x227 center, and convert it to BGR
        dim = self.get_input_dim()
        image = util.crop_image_center(img)
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
        try:
            last_layer = self.net_.caffenet.blobs.items()[-1]
            num_output = len(last_layer[1].data.flatten())
        except: # it means you have the old version of caffe
            num_output = 1000
        output_blobs = [np.empty((num, num_output, 1, 1), dtype=np.float32)]
        self.net_.caffenet.Forward(input_blob, output_blobs)
        net_representation = {k: output for k, output in \
                                    self.net_.caffenet.blobs.items()}
        return net_representation

    def visualize_features(self, net_representation, layers = [], \
                                fig_handle = 0, subsample_kernels = 4,\
                                dump_image_path = '', string_drop=''):
        plt.rcParams['image.interpolation'] = 'nearest'
        i = 1
        if layers == []:
            layers = net_representation.keys()
        for layer in layers:
            feat  = net_representation[layer].data[0].copy()
            fig_handle.add_subplot(3,len(layers),len(layers)+i)
            plt.title(layer)
            if layer == 'data':
                # input
                image = feat
                image -= image.min()
                image /= image.max()
                self.showimage_(image.transpose(1, 2, 0))
                if not string_drop=='':
                    plt.text(-10, 300, string_drop, backgroundcolor='white')
            elif 'fc' in layer or 'prob' in layer:
                # full connections
                plt.plot(feat.flat)
                plt.xticks([0, len(feat.flat)])
                plt.yticks([np.ceil(np.min(feat.flat)), \
                                        np.floor(np.max(feat.flat))])
            else: # 'conv' in layer or 'pool' in layer or 'norm' in layer
                # conv layers
                self.vis_square_(feat[::subsample_kernels,:,:], padval=1)
            i += 1
        if not dump_image_path == '':
            plt.savefig(dump_image_path + '.eps', dpi = 100,\
                                        bbox_inches = 'tight')


    def showimage_(self, im):
        """
        The network takes BGR images, so we need to switch
        color channels
        """
        if im.ndim == 3:
            im = im[:, :, ::-1]
        plt.imshow(im)
        plt.axis('off')

    def vis_square_(self, data, padsize=1, padval=0):
        """
        Take an array of shape (n, height, width) or (n, height, width,
        channels) and visualize each (height, width) thing in a grid of
        size approx. sqrt(n) by sqrt(n)
        """
        data -= data.min()
        data /= data.max()
        # force the number of filters to be square
        n = int(np.ceil(np.sqrt(data.shape[0])))
        padding = ((0, n ** 2 - data.shape[0]), (0, padsize), (0, padsize)) \
                    + ((0, 0),) * (data.ndim - 3)
        data = np.pad(data, padding, mode='constant', \
                                constant_values=(padval, padval))
        # tile the filters into an image
        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) \
                    + tuple(range(4, data.ndim + 1)))
        data = data.reshape((n * data.shape[1], n * data.shape[3]) \
                    + data.shape[4:])
        self.showimage_(data)
