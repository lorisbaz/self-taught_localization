from util import *
from configuration import *
from featextractor import *

output_dir = 'featextractor_specs'

def create000():
    conf = Configuration()
    netparams = NetworkCaffeParams( \
                       conf.ilsvrc2012_caffe_model_spec, \
                       conf.ilsvrc2012_caffe_model, \
                       conf.ilsvrc2012_caffe_wnids_words, \
                       conf.ilsvrc2012_caffe_avg_image, \
                       center_only=True)
    feature_extractor_params =  FeatureExtractorNetworkParams(netparams,\
                                layer='fc7', cache_features=True)
    fname = output_dir + '/000.pkl'
    dump_obj_to_file_using_pickle(feature_extractor_params, fname)

def create001():
    # USE GIRSHICK MODEL
    conf = Configuration(load_Girshick_caffe_model = True)
    netparams = NetworkCaffeParams( \
                       conf.ilsvrc2012_caffe_model_spec, \
                       conf.ilsvrc2012_caffe_model, \
                       conf.ilsvrc2012_caffe_wnids_words, \
                       conf.ilsvrc2012_caffe_avg_image, \
                       center_only=True)
    feature_extractor_params =  FeatureExtractorNetworkParams(netparams,\
                                layer='fc7', cache_features=True)
    fname = output_dir + '/001.pkl'
    dump_obj_to_file_using_pickle(feature_extractor_params, fname)


if __name__ == '__main__':
    #create000()
    create001()
