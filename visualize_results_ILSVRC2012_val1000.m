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
title('Results on ILSVRC2012-val-1000')

% create the figure for the MABO score
figure;
h_mean_mabo = gca;
hold on;
grid on;
axis([1, 70, 0, 1]);
xlabel('Num subwindows')
ylabel('MABO')
title('Results on ILSVRC2012-val-1000')

% create the figure for the Precision
figure;
h_precision = gca;
hold on;
grid on;
axis([1, 70, 0, 0.5]);
xlabel('Num subwindows')
ylabel('Precision')
title('Results on ILSVRC2012-val-1000')

% *** our experiments
% this is list of cells of 2-elements-cells {experiment_name, legend}
params.prefix_output_files = 'results_ILSVRC2012val1000';
params.exps = {{'exp06_16stats','exp06_16 (GrayBox, topC=5)'}, ...
               {'exp14_06stats','exp14_06 (SelectiveSearch, fast)'}, ...
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
