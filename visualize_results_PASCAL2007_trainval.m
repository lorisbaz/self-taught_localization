function visualize_results_PASCAL2007_trainval()
% This script plots results for PASCAL VOC 2007 - TRAINVAL set.
%
%

% clear the variables
clear;

% load common plot definitions
plot_defs;

% parameters
params.exp_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.save_output_files = 1;
params.set_log_scale = 1;
params.prefix_output_files = 'results_PASCAL2007trainval';

% create the figure for the mean recall per class
figure;
h_mean_recall = gca;
hold on;
grid on;
axis([1, 70, 0, 1]);
xlabel('Num subwindows')
ylabel('Mean recall per class')
title('Results on PASCAL-2007-trainval')

% create the figure for the MABO score
figure;
h_mean_mabo = gca;
hold on;
grid on;
axis([1, 70, 0, 1]);
xlabel('Num subwindows')
ylabel('MABO')
title('Results on PASCAL-2007-trainval')

% create the figure for the Precision
figure;
h_precision = gca;
hold on;
grid on;
axis([1, 70, 0, 0.5]);
xlabel('Num subwindows')
ylabel('Precision')
title('Results on PASCAL-2007-trainval')

% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
% params.exps = {{'exp06_13stats','exp06_13 (GrayBox, topC=5)'}, ...
%                {'exp06_14stats', 'exp06_14 (SlidingWindow, topC=5)'}, ...
%                {'exp06_15stats', 'exp06_15 (GraySegm, topC=5)'}, ...
%                {'exp06_17stats', 'exp06_17 (GrayBox, topC=5, quantile_pred=0.98)'}, ...
%                {'exp06_18stats', 'exp06_18 (GrayBox, topC=20, quantile_pred=0.99, minTopC=5'}, ...
%                {'exp14_04stats', 'exp14_04 (SelectiveSearch, fast)'}, ...               
%                {'exp21_02stats', 'exp21_02 (GrayBox+GraySegm, topC=5)'}, ...
%                {'exp22_03stats', 'exp22_03 (Re-ranked GrayBox, topC=5)'}, ...
%                {'exp22_04stats', 'exp22_04 (Re-ranked GrayBox+GraySegm, topC=5)'}, ...
%                };

params.exps = {{'exp06_26stats_NMS_05','exp06_26_NMS_05 (GrayBox, topC=5)'}, ...
               {'exp06_27stats_NMS_05', 'exp06_27_NMS_05 (GraySegm, topC=5)'}, ...
               {'exp14_08stats', 'exp14_08 (SelectiveSearch, fast)'}, ... 
               {'exp23_08stats_NMS_05', 'exp23_08_NMS_05 (ObfuscationSearch, topC=5)'}, ...
               };
% params.exps = {{'exp06_13stats_NMS_05','exp06_13_NMS_05 (GrayBox, topC=5)'}, ...
%                {'exp06_25stats_NMS_05', 'exp06_25_NMS_05 (GraySegm, topC=5)'}, ...
%                {'exp14_04stats', 'exp14_04 (SelectiveSearch, fast)'}, ...              
%                {'exp23_07stats_NMS_05', 'exp23_07_NMS_05 (ObfuscationSearch, topC=5)'}, ...
%                };
           
           
for i=1:numel(params.exps)
  % load the experiment results
  S=load([params.exp_dir '/' params.exps{i}{1} '/mat/recall_vs_numPredBboxesImage.mat']);  
  % plot the mean recall per class
  [X, Y] = cut_tail_with_equal_values(S.x_values, S.mean_recall);
  plot(h_mean_recall, X, Y, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  % plot the MABO
  [X, Y] = cut_tail_with_equal_values(S.x_values, S.mean_ABO);
  plot(h_mean_mabo, X, Y, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  % plot the Precision
  [X, Y] = cut_tail_with_equal_values(S.x_values, S.precision);
  plot(h_precision, X, Y, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_precision, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
end
x_values = S.x_values;     

% *** SS (from BING paper)
if 0
S=load('plot_defs_Cheng_CVPR14.mat');
plot(h_mean_recall, x_values, S.SS_IJCV13(x_values), '-o', 'DisplayName', 'SS_IJCV13', 'Color', MATLAB.Color.green);
h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
end

% *** BING BING_RGB_HSV_Gray_CVPR14
if 0
% load the results
S=load('plot_defs_Cheng_CVPR14.mat');
% plot the mean recall per class
plot(h_mean_recall, x_values, S.BING_RGB_HSV_Gray_CVPR14(x_values), '-o', 'DisplayName', 'BING_RGB_HSV_Gray_CVPR14', 'Color', MATLAB.Color.greenDark);
h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
% plot the MABO
plot(h_mean_mabo, x_values, S.MABO(x_values), '-o', 'DisplayName', 'BING (?)', 'Color', MATLAB.Color.greenDark);
h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
end

if params.set_log_scale
    set(h_mean_recall, 'XScale', 'log')
    set(h_mean_mabo, 'XScale', 'log')
    set(h_precision, 'XScale', 'log')
end

% *** save figures
if params.save_output_files
  saveas(h_mean_recall, [params.prefix_output_files '_mean_recall.png']);
  saveas(h_mean_mabo, [params.prefix_output_files '_mean_mabo.png'])
  saveas(h_precision, [params.prefix_output_files '_precision.png'])
end

end