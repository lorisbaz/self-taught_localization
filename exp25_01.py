from util import *

from detector import *
from featextractor import *
from network import *
from configuration import *
from pipeline_detector import *

if __name__ == "__main__":
    conf = Configuration()    
    # *** FeatureExtractor
    # TODO. Decide with Loris how to make sure the info is consistent.
    #       Also, does it work on the cluster? Maybe we should create a factory
    #       for the class Network as well.
    net = NetworkCaffe(conf.ilsvrc2012_caffe_model_spec, \
                       conf.ilsvrc2012_caffe_model, \
                       conf.ilsvrc2012_caffe_wnids_words, \
                       conf.ilsvrc2012_caffe_avg_image, \
                       center_only=True)
    feature_extractor_params =  FeatureExtractorNetworkParams(network=net,\
                                layer='fc7', cache_features=False)
    # *** Detector
    detector_params = DetectorLinearSVMParams()
    # *** PipelineDetectorParams
    params = PipelineDetectorParams()
    # experiment name
    params.exp_name = 'exp25_01'
    # input
    params.exp_name_input = 'exp24_04'
    # categories, splits
    params.categories_file = 'pascal2007/categories.txt'
    params.splits_dir = 'pascal2007'
    params.split_train_name = 'trainval'
    # input/output dirs
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name_input 
    # FeatureExtractor module to use (parameters object)
    params.feature_extractor_params = feature_extractor_params
    # Detector module to use (parameters object)
    params.detector_params = detector_params
    # field_name_for_pred_objects_in_AnnotatedImage
    params.field_name_for_pred_objects_in_AnnotatedImage = 'SELECTIVESEARCH'
    # run on Anthill?
    params.run_on_anthill = False
    # run just the first category
    params.categories_to_process = [0]
    # *** run the pipeline
    PipelineDetector.train_evaluate_detectors(params)
