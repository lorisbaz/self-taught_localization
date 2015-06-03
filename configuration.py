"""
Configuration class to store paths.
"""

class Configuration:
    def __init__(self,root):
        self.caffe_model_spec = \
                    './prototxt/imagenet_deploy_GRAYOBFUSCATION.prototxt'
        self.caffe_model = root + '/caffe_reference_imagenet_model'
        self.caffe_avg_image = root + '/ilsvrc_2012_mean.npy'
        self.caffe_wnids_words = './prototxt/synset_words.txt'
