function visualize_results_ILSVRC2012_val200rnd()
% This script plots results for ILSVRC 2012 - validation (200 classes).
%
%

% clear the variables
clear;

% load common plot definitions
plot_defs;

% parameters
params.exp_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.dataset_name = 'ILSVRC2012val200rnd';
params.prefix_output_files = strcat('results_',params.dataset_name);
params.save_output_files = 1;
params.set_log_scale = 1;
params.mean_precision = 1;
params.MATLAB = MATLAB;

% *** our experiments
% params.exps is list of cells of 2-elements-cells {experiment_name, legend}

% SOME STANDARD EXPERIMENTS

% params.exps = {{'exp06_19stats','exp06_19 (GrayBox, topC=5)'}, ...
%                {'exp06_20stats','exp06_20 (GraySegm, topC=5)'}, ...
%                {'exp21_03stats','exp21_03 (GraySegm+GrayBox, topC=5)'}, ...
%                };
          
params.exps = {{'exp06_19stats_NMS_05','exp06_19_NMS_05 (GrayBox, topC=5)'}, ...
              {'exp06_30stats_NMS_05','exp06_30_NMS_05 (GrayBox, GT label)'}, ...
              {'exp06_20stats_NMS_05','exp06_20_NMS_05 (GraySegm, topC=5)'}, ...
              {'exp06_29stats_NMS_05','exp06_29_NMS_05 (GraySegm, GT label)'}, ...
              {'exp23_03stats_NMS_05','exp23_03_NMS_05 (ObfuscationSearch, topC=5)'}, ...
              {'exp23_10stats_NMS_05','exp23_10_NMS_05 (ObfuscationSearch, GT label)'}, ...
              };
             
% params.exps = {{'exp06_21stats','exp06_21 (SlidingWindow, topC=5)'}, ...
%                {'exp06_21stats_NMS_05','exp06_21_NMS_05 (SlidingWindow, topC=5)'}, ...
%                {'exp06_21stats_NMS_09','exp06_21_NMS_09 (SlidingWindow, topC=5)'}, ...
%                };

visualize_plot_and_save(params);

  
end

