function visualize_results_ILSVRC2012_val200()
% This script plots results for ILSVRC 2012 - validation (200 classes).
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
title('Results on ILSVRC2012-val-200')

% create the figure for the MABO score
figure;
h_mean_mabo = gca;
hold on;
grid on;
axis([1, 50, 0, 1]);
xlabel('Num subwindows')
ylabel('MABO')
title('Results on ILSVRC2012-val-200')

% create the figure for the Precision
figure;
h_precision = gca;
hold on;
grid on;
axis([1, 50, 0, 0.5]);
xlabel('Num subwindows')
ylabel('Precision')
title('Results on ILSVRC2012-val-200')

% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
params.exps = {{'exp06_10stats','exp06_10 (GrayBox, topC=5)'}, ...
               {'exp06_12stats','exp06_12 (GraySegm, topC=5)'}, ...
               {'exp06_11stats','exp06_11 (SlidingWindow, topC=5)'}, ...
               {'exp14_02stats', 'exp14_02 (SelectiveSearch, fast)'}, ...
               {'exp22_01stats', 'exp22_01 (Re-rank GrayBox)'}, ...
               {'exp22_02stats', 'exp22_02 (Re-rank GraySegm+GrayBox)'}, ...             
               {'exp23_01stats', 'exp23_01 (Obfuscation Seach)'}, ...
               };

for i=1:numel(params.exps)
  % load the experiment results
  S=load([params.exp_dir '/' params.exps{i}{1} '/mat/recall_vs_numPredBboxesImage.mat']);  
  % plot the mean recall per class
  plot(h_mean_recall, S.x_values, S.mean_recall, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  % plot the MABO
  plot(h_mean_mabo, S.x_values, S.mean_ABO, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  % plot the Precision
  plot(h_precision, S.x_values, S.precision, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
  h=legend(h_precision, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
end


% *** save figures
saveas(h_mean_recall, 'results_ILSVRC2012val200_mean_recall.png');
saveas(h_mean_mabo, 'results_ILSVRC2012val200_mean_mabo.png')
saveas(h_precision, 'results_ILSVRC2012val200_precision.png')

end
