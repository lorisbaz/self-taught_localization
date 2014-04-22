function visualize_plot_and_save(params)
           
MATLAB = params.MATLAB;
try 
    if params.PASCAL2007_test
        fprintf('Plot also results from BING CVPR 14 paper.\n');
    end
catch
    params.PASCAL2007_test = 0;
end

% create the figure for the mean recall per class
figure;
h_mean_recall = gca;
hold on;
grid on;
axis([1, 50, 0, 1]);
xlabel('Num subwindows')
ylabel('Mean recall per class')
title(strcat('Results on ', params.dataset_name))

% create the figure for the MABO score
figure;
h_mean_mabo = gca;
hold on;
grid on;
axis([1, 50, 0, 1]);
xlabel('Num subwindows')
ylabel('MABO')
title(strcat('Results on ', params.dataset_name))

% create the figure for the Precision
figure;
h_precision = gca;
hold on;
grid on;
axis([1, 50, 0, 0.6]);
xlabel('Num subwindows')
ylabel('Precision')
title(strcat('Results on ', params.dataset_name))

if params.mean_precision
    % create the figure for the Precision
    figure;
    h_mean_precision = gca;
    hold on;
    grid on;
    axis([1, 50, 0, 0.6]);
    xlabel('Num subwindows')
    ylabel('Mean Precision')
    title(strcat('Results on ', params.dataset_name))
end

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
  % plot the Precision    
  if params.mean_precision
      [X, Y] = cut_tail_with_equal_values(S.x_values, S.mean_precision);
      plot(h_mean_precision, X, Y, '-', 'DisplayName', params.exps{i}{2}, 'Color', MATLAB.Colors_all{i}, 'Marker', MATLAB.LineSpec.markers(i));
      h=legend(h_mean_precision, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
  end
end

% ---------- only PASCAL/testset!!! ---------- %
x_values = S.x_values;     
% *** SS (from BING paper)
if params.PASCAL2007_test
    S=load('plot_defs_Cheng_CVPR14.mat');
    plot(h_mean_recall, x_values, S.SS_IJCV13(x_values), '-o', 'DisplayName', 'SS_IJCV13', 'Color', MATLAB.Color.green);
    h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
end
% *** BING BING_RGB_HSV_Gray_CVPR14
if params.PASCAL2007_test
    % load the results
    S=load('plot_defs_Cheng_CVPR14.mat');
    % plot the mean recall per class
    plot(h_mean_recall, x_values, S.BING_RGB_HSV_Gray_CVPR14(x_values), '-o', 'DisplayName', 'BING_RGB_HSV_Gray_CVPR14', 'Color', MATLAB.Color.greenDark);
    h=legend(h_mean_recall, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
    % plot the MABO
    plot(h_mean_mabo, x_values, S.MABO(x_values), '-o', 'DisplayName', 'BING (?)', 'Color', MATLAB.Color.greenDark);
    h=legend(h_mean_mabo, '-DynamicLegend'); set(h,'Interpreter','none', 'Location', 'Best');
end
% ------- end - only PASCAL/testset!!! ------- %


% Log scale
if params.set_log_scale
    set(h_mean_recall, 'XScale', 'log')
    set(h_mean_mabo, 'XScale', 'log')
    set(h_precision, 'XScale', 'log')
    if params.mean_precision
        set(h_mean_precision, 'XScale', 'log')
    end
end

% *** save figures
if params.save_output_files
  saveas(h_mean_recall, [params.prefix_output_files '_mean_recall.png']);
  saveas(h_mean_mabo, [params.prefix_output_files '_mean_mabo.png'])
  saveas(h_precision, [params.prefix_output_files '_precision.png'])
  if params.set_log_scale
    saveas(h_mean_precision, [params.prefix_output_files '_precision.png'])
  end
end

end