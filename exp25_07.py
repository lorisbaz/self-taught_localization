import vlg.util.pbar
import vlg.util.parfun

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
                                       'featextractor_specs/001.pkl')
    # *** Detector
    detector_params = DetectorLinearSVMParams()
    # *** PipelineDetectorParams
    params = PipelineDetectorParams()
    # experiment name
    params.exp_name = 'exp25_07'
    # input
    params.exp_name_input_train = 'exp24_11'
    params.exp_name_input_test = 'exp24_12'
    # categories, splits
    params.categories_file = 'ILSVRC2013_clsloc/200RND/categories.txt'
    params.splits_dir = 'ILSVRC2013_clsloc/200RND/'
    params.split_train_name = 'train'
    params.split_test_name = 'val'
    # number of iterations
    params.negative_mining = 'batch'
    params.num_iterations = 2
    # perform the evaluation only in the last iteration
    params.iterations_to_evaluate = [params.num_iterations - 1]
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
    # field names for the pos/neg bboxes
    params.field_name_pos_bboxes = 'GT'
    params.field_name_bboxes = 'PRED:SELECTIVESEARCH'
    # neg_bboxes_overlapping_with_pos_params
    params.neg_bboxes_overlapping_with_pos_params = [0.0, 0.3, 0.3, 0.7]
    # visualization
    params.progress_bar_params = vlg.util.pbar.ProgressBarPlusParams()
    # ParFun Categories
    parfun_tmpdir = \
            '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/TEMP2'
    if 1:  # -- Anthill
        params.parfun_params_categories = vlg.util.parfun.ParFunAnthillParams( \
                        time_requested=24, memory_requested=6, \
                        progress_bar_params = params.progress_bar_params, \
                        tmp_dir = parfun_tmpdir, max_tasks=100)
    if 0:  # -- Local
        params.parfun_params_categories = vlg.util.parfun.ParFunDummyParams()
    # ParFun TRAINING
    if 0:  # -- Anthill
        params.parfun_params_training = vlg.util.parfun.ParFunAnthillParams( \
                        time_requested=10, memory_requested=2, \
                        progress_bar_params = params.progress_bar_params, \
                        tmp_dir = parfun_tmpdir, max_tasks=1000)
    if 1:  # -- Local, multi-core
        params.parfun_params_training = vlg.util.parfun.ParFunProcessesParams( \
                num_processes = 12, \
                progress_bar_params = params.progress_bar_params)
    if 0:  # -- Local
        params.parfun_params_training = vlg.util.parfun.ParFunDummyParams()
    # ParFun EVALUATION
    if 0:  # -- Anthill
        params.parfun_params_evaluation = vlg.util.parfun.ParFunAnthillParams( \
                        time_requested=10, memory_requested=2, \
                        progress_bar_params = params.progress_bar_params, \
                        tmp_dir = parfun_tmpdir, max_tasks=1000)
    if 1:  # -- Local, multi-core
        params.parfun_params_evaluation = vlg.util.parfun.ParFunProcessesParams( \
                num_processes = 12, \
                progress_bar_params = params.progress_bar_params)
    if 0:  # -- Local
        params.parfun_params_evaluation = vlg.util.parfun.ParFunDummyParams()
    # run just the first category
    params.categories_to_process = []
    # *** run the pipeline
    PipelineDetector.train_evaluate_detectors(params)
