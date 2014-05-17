function visualize_results_ILSVRC2013det_train150()
% This script plots results for ILSVRC 2013 detection - training (150 classes).
%
%

% clear the variables
clear;

% load common plot definitions
plot_defs;

% parameters
params.exp_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.dataset_name = 'ILSVRC2013det_train200rnd';
params.prefix_output_files = strcat('results_',params.dataset_name);
params.save_output_files = 1;
params.set_log_scale = 1;
params.mean_precision = 1;
params.MATLAB = MATLAB;

% *** our experiments
% params.exps is list of cells of 2-elements-cells {experiment_name, legend}

params.exps = {{'exp23_13stats_NMS_05', 'exp23_13stats_NMS_05 (ObfuscationSearch, Top5)'}, ...
               {'exp23_15stats_NMS_05', 'exp23_15stats_NMS_05 (ObfuscationSearch, GT)'}, ...
               };


visualize_plot_and_save(params);
  
end

