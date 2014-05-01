import vlg.util.pbar

from util import *
from detector import *
from featextractor import *
from network import *
from configuration import *
from pipeline_detector import *

if __name__ == "__main__":
    conf = Configuration()    
    # *** FeatureExtractor
    feature_extractor_params = load_obj_from_file_using_pickle( \
                                       'featextractor_specs/000.pkl')    
    # *** Detector
    detector_params = DetectorLinearSVMParams()
    # *** PipelineDetectorParams
    params = PipelineDetectorParams()
    # experiment name
    params.exp_name = 'exp25_03'
    # input
    params.exp_name_input_train = 'exp24_04'
    params.exp_name_input_test = 'exp24_02'
    # categories, splits
    params.categories_file = 'pascal2007/categories.txt'
    params.splits_dir = 'pascal2007'
    params.split_train_name = 'trainval'
    params.split_test_name = 'test'
    # input/output dirs
    params.output_dir = conf.experiments_output_directory \
                        + '/' + params.exp_name
    params.input_dir_train = conf.experiments_output_directory \
                        + '/' + params.exp_name_input_train
    params.input_dir_test = conf.experiments_output_directory \
                        + '/' + params.exp_name_input_test                        
    # FeatureExtractor module to use (parameters object)
    params.feature_extractor_params = feature_extractor_params
    # Detector module to use (parameters object)
    params.detector_params = detector_params
    # field_name_for_pred_objects_in_AnnotatedImage
    params.field_name_for_pred_objects_in_AnnotatedImage = 'SELECTIVESEARCH'
    # visualization
    self.progress_bar_params vlg.util.pbar.ProgressBarPlusParams()
    # ParFun Categories
    parfun_tmpdir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/TEMP')
    if 0:
        self.parfun_params_categories = ParFunAnthillParams( \
                        time_requested=10, memory_requested=6, \
                        progress_bar_params = self.progress_bar_params, \
                        tmp_dir = parfun_tmpdir
    if 1:
        self.parfun_params_categories = ParFunDummyParams()
    # ParFun TRAINING        
    if 1:
        self.parfun_params_training = ParFunAnthillParams( \
                        time_requested=10, memory_requested=2, \
                        progress_bar_params = self.progress_bar_params, \
                        tmp_dir = parfun_tmpdir
    if 0:
        self.parfun_params_training = ParFunProcessesParams( \
                num_processes = 8, \
                progress_bar_params = self.progress_bar_params)                        
    # ParFun EVALUATION                
    if 1:
        self.parfun_params_evaluation = ParFunAnthillParams( \
                        time_requested=10, memory_requested=2, \
                        progress_bar_params = self.progress_bar_params, \
                        tmp_dir = parfun_tmpdir
    if 0:
        self.parfun_params_evaluation = ParFunProcessesParams( \
                num_processes = 8, \
                progress_bar_params = self.progress_bar_params)
                    
    # run just the first category
    params.categories_to_process = [0]
    # *** run the pipeline
    PipelineDetector.train_evaluate_detectors(params)
