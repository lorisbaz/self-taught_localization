function visualize_results_ILSVRC2012_val_comparison_subsets()
% This script plots results for ILSVRC 2012 - validation.
%
%

% clear the variables
clear;

% load common plot definitions
plot_defs;

% parameters
params.exp_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.dataset_name = 'ILSVRC2012valComparisonSubsets_mean_recall';
params.prefix_output_files = strcat('results_',params.dataset_name);
params.save_output_files = 1;
params.set_log_scale = 1;
params.mean_precision = 1;
params.MATLAB = MATLAB;

% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
params.exps = {{'exp06_10stats','exp06_10 (first 200 classes) (GrayBox, topC=5)'}, ...
               {'exp06_16stats','exp06_16 (all 1000 classes) (GrayBox, topC=5)'}, ...
               {'exp06_19stats','exp06_19 (random 200 classes) (GrayBox, topC=5)'}, ...
               };
           
visualize_plot_and_save(params);

end
