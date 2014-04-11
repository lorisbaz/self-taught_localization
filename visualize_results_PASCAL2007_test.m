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
params.save_output_files = 1;
params.set_log_scale = 1;
params.eps_thr = 0.001; % do not visualize the curve when the improvement is less than eps_thr

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
axis([1, 70, 0, 1]);
xlabel('Num subwindows')
ylabel('MABO')
title('Results on PASCAL-2007-test')

% create the figure for the Precision
figure;
h_precision = gca;
hold on;
grid on;
axis([1, 70, 0, 0.5]);
xlabel('Num subwindows')
ylabel('Precision')
title('Results on PASCAL-2007-test')

% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
params.prefix_output_files = 'results_PASCAL2003test';
params.exps = {{'exp06_13stats','exp06_13 (GrayBox, topC=5)'}, ...
               {'exp06_14stats', 'exp06_14 (SlidingWindow, topC=5)'}, ...
               {'exp06_15stats', 'exp06_15 (GraySegm, topC=5)'}, ...
               {'exp06_17stats', 'exp06_17 (GrayBox, topC=5, quantile_pred=0.98)'}, ...
               {'exp06_18stats', 'exp06_18 (GrayBox, topC=20, quantile_pred=0.99, minTopC=5'}, ...
               {'exp14_04stats', 'exp14_04 (SelectiveSearch, fast)'}, ...               
               {'exp21_02stats', 'exp21_02 (GrayBox+GraySegm, topC=5)'}, ...
               {'exp22_03stats', 'exp22_03 (Re-ranked GrayBox, topC=5)'}, ...
               {'exp22_04stats', 'exp22_04 (Re-ranked GrayBox+GraySegm, topC=5)'}, ...
               };

for i=1:numel(params.exps)
  % load the experiment results
  S=load([params.exp_dir '/' params.exps{i}{1} '/mat/recall_vs_numPredBboxesImage.mat']);  
  % plot the mean recall per class
  [S.x_values2, S.mean_recall2] = cut_off_curve(S.x_values, S.mean_recall, params.eps_thr);
  plot(h_mean_recall, S.x_values2, S.mean_recall2, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  % plot the MABO
  [S.x_values2, S.mean_ABO2] = cut_off_curve(S.x_values, S.mean_ABO, params.eps_thr);
  plot(h_mean_mabo, S.x_values2, S.mean_ABO2, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  % plot the Precision
  [S.x_values2, S.precision2] = cut_off_curve(S.x_values, S.precision, params.eps_thr);
  plot(h_precision, S.x_values2, S.precision2, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_precision, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
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
