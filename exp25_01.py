from util import *

from detector import *
from featextractor import *
from network import *
from configuration import *
from pipeline_detector import *

if __name__ == "__main__":
    # *** PipelineDetectorParams
    conf = Configuration()
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
    params.feature_extractor_params = None
    # Detector module to use (parameters object)
    params.detector_params = None
    # run on Anthill?
    params.run_on_anthill = False
    # run just the first category
    params.categories_to_process = [0]
    # *** run the pipeline
    train_evaluate_detectors(params)
