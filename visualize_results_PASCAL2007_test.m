function visualize_results_PASCAL2007_test()
% This script plots results for PASCAL VOC 2007 - TEST set.
%
%

% clear the variables
clear;

% load common plot definitions
plot_defs;

% parameters
params.exp_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';

% create the figure for the mean recall per class
figure;
h_mean_recall = gca;
hold on;
grid on;
axis([1, 70, 0, 1]);
xlabel('Num subwindows')
ylabel('Mean recall per class')
title('Results on PASCAL-2007-test')

% create the figure for the MABO score
figure;
h_mean_mabo = gca;
hold on;
grid on;
axis([1, 50, 0, 1]);
xlabel('Num subwindows')
ylabel('MABO')
title('Results on PASCAL-2007-test')

% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
params.exps = {{'exp06_13stats','exp06_13 (GrayBox, topC=5)'}, ...
               {'exp06_14stats', 'exp06_14 (SlidingWindow, topC=5)'}, ...
               {'exp06_15stats', 'exp06_15 (GraySegm, topC=5)'}, ...
               {'exp06_17stats', 'exp06_17 (GrayBox, topC=5, quantile_pred=0.98)'}, ...
               };

for i=1:numel(params.exps)
  % load the experiment results
  S=load([params.exp_dir '/' params.exps{i}{1} '/mat/recall_vs_numPredBboxesImage.mat']);  
  % plot the mean recall per class
  plot(h_mean_recall, S.x_values, S.mean_recall, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none');
  % plot the MABO
  plot(h_mean_mabo, S.x_values, S.mean_ABO, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none');
end
     

% *** SS (from BING paper)
if 1
S=load('plot_defs_Cheng_CVPR14.mat');
plot(h_mean_recall, [1:numel(S.SS_IJCV13)], S.SS_IJCV13, '-o', 'DisplayName', 'SS_IJCV13', 'Color', MATLAB.Color.green);
h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none');
end

% *** BING BING_RGB_HSV_Gray_CVPR14
if 1
% load the results
S=load('plot_defs_Cheng_CVPR14.mat');
% plot the mean recall per class
plot(h_mean_recall, [1:numel(S.BING_RGB_HSV_Gray_CVPR14)], S.BING_RGB_HSV_Gray_CVPR14, '-o', 'DisplayName', 'BING_RGB_HSV_Gray_CVPR14', 'Color', MATLAB.Color.greenDark);
h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none');
% plot the MABO
plot(h_mean_mabo, [1:numel(S.MABO)], S.MABO, '-o', 'DisplayName', 'BING (?)', 'Color', MATLAB.Color.greenDark);
h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none');
end

end
