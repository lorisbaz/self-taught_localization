from util import *

from detector import *
from featextractor import *
from network import *
from configuration import *
from pipeline_detector_online import *

if __name__ == "__main__":
    conf = Configuration()
    # *** FeatureExtractor
    feature_extractor_params = load_obj_from_file_using_pickle( \
                                       'featextractor_specs/001.pkl')
    # *** Detector
    detector_params = DetectorLinearSVMParams()
    detector_params.Call = [1e-3]
    detector_params.B = 10
    class_weight = {1:2.0, -1:1.0}
    # *** PipelineDetectorParams
    params = PipelineDetectorParams()
    # experiment name
    params.exp_name = 'exp25_08_C=1e-3_SCALE=0.3_NEGS=2000'
    # input
    params.exp_name_input_train = 'exp24_07'
    params.exp_name_input_test = 'exp24_06'
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
    # FeatProcessorParams
    params.feature_processor_params = FeatProcessorScaleParams(0.30)
    # Detector module to use (parameters object)
    params.detector_params = detector_params
    # field names for the pos/neg bboxes
    params.field_name_pos_bboxes = 'GT'
    params.field_name_bboxes = 'PRED:SELECTIVESEARCH'
    params.negatives_threshold_confidence_single_image = -1.0001
    params.negatives_threshold_confidence_entire_set = 1.2
    # visualization
    params.progress_bar_params = vlg.util.pbar.ProgressBarPlusParams()
    # ParFun Categories
    parfun_tmpdir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/TEMP'
    if 0:  # -- Anthill
        params.parfun_params_categories = vlg.util.parfun.ParFunAnthillParams( \
                        time_requested=10, memory_requested=6, \
                        progress_bar_params = params.progress_bar_params, \
                        tmp_dir = parfun_tmpdir, max_tasks=100)
    if 1:  # -- Local
        params.parfun_params_categories = vlg.util.parfun.ParFunDummyParams()
    # ParFun TRAINING
    if 0:  # -- Anthill
        params.parfun_params_training = vlg.util.parfun.ParFunAnthillParams( \
                        time_requested=10, memory_requested=2, \
                        progress_bar_params = params.progress_bar_params, \
                        tmp_dir = parfun_tmpdir, max_tasks=1000)
    if 0:  # -- Local, multi-core
        params.parfun_params_training = vlg.util.parfun.ParFunProcessesParams( \
                num_processes = 8, \
                progress_bar_params = params.progress_bar_params)
    if 1:  # -- Local
        params.parfun_params_training = vlg.util.parfun.ParFunDummyParams()
    # ParFun EVALUATION
    if 1:  # -- Anthill
        params.parfun_params_evaluation = vlg.util.parfun.ParFunAnthillParams( \
                        time_requested=10, memory_requested=2, \
                        progress_bar_params = params.progress_bar_params, \
                        tmp_dir = parfun_tmpdir, max_tasks=1000)
    if 0:  # -- Local, multi-core
        params.parfun_params_evaluation = vlg.util.parfun.ParFunProcessesParams( \
                num_processes = 12, \
                progress_bar_params = params.progress_bar_params)
    if 0:  # -- Local
        params.parfun_params_evaluation = vlg.util.parfun.ParFunDummyParams()
    # run just the first category
    params.categories_to_process = [0]
    # *** run the pipeline
    PipelineDetector.train_evaluate_detectors(params)
