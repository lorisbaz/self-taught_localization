function visualize_results_ILSVRC2012_val1000()
% This script plots results for ILSVRC 2012 - validation (1000 classes).
%
%

% clear the variables
clear;

% load common plot definitions
plot_defs;

% parameters
params.exp_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.dataset_name = 'results_ILSVRC2012val1000';
params.prefix_output_files = strcat('results_',params.dataset_name);
params.save_output_files = 1;
params.set_log_scale = 1;
params.mean_precision = 1;
params.MATLAB = MATLAB;


% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
params.prefix_output_files = 'results_ILSVRC2012val1000';
% params.exps = {{'exp06_16stats','exp06_16 (GrayBox, topC=5)'}, ...
%                {'exp06_16stats_NMS_05','exp06_16_NMS_05 (GrayBox, topC=5)'}, ...
%                {'exp06_16stats_NMS_09','exp06_16_NMS_09 (GrayBox, topC=5)'}, ...
%                {'exp14_06stats','exp14_06 (SelectiveSearch, fast)'}, ...
%                };
params.exps = { ...
               {'exp30_10stats_NMS_05','ObfuscationSearch similarity + NET features, topC=5'}, ...
               };

visualize_plot_and_save(params);

end
