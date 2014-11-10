%% Analysis of the results varying the alphas
results_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
input_dir = [results_dir '/exp30_02_alphas'];
mat_dir = [input_dir 'stats_NMS_05'];

% top-M alphas
topM = 10;

% load list of alphas
load([input_dir '/list_of_alphas.mat'])

% Read results from disk
results = cell(1, size(alphas, 2));
not_working = [];
for i = 1:size(alphas, 2)
  foldername = sprintf('%05d', i-1);
  try
    results{i} = load([mat_dir '/' foldername '/mat/recall_vs_numPredBboxesImage.mat']);
  catch
    fprintf('File not found: %s \n', [mat_dir '/' foldername '/mat/recall_vs_numPredBboxesImage.mat'])
    not_working = [not_working, i];
  end
end

% Accumulate the statistics to evaluate the results
stats.pre_rec = zeros(1, size(alphas, 2));
stats.pre_rec_area = zeros(1, size(alphas, 2));
stats.mean_precision = zeros(1, size(alphas, 2));
stats.mean_recall = zeros(1, size(alphas, 2));
for i = 1:size(alphas, 2)
  try 
    stats.pre_rec(i) = results{i}.mean_precision(1) + results{i}.mean_recall(1);
    stats.pre_rec_area(i) = (sum(results{i}.mean_precision) + sum(results{i}.mean_recall))/length(results{i}.mean_precision);
    stats.mean_precision(i) = results{i}.mean_precision(1);
    stats.mean_recall(i) = results{i}.mean_recall(1);
    %stats.mean_AP(i) = results{i}.mean_average_precision(1);
  catch
    fprintf('Alpha %d not used \n', i);
  end
end


% plot top-M
[values, idx] = sort(stats.pre_rec, 'descend');
[values(1:topM); alphas(:, idx(1:topM))] % Order: obfuscation, size, fill, features
idx = idx(values~=0);

% Visualize
figure;
subplot(131)
plot(stats.pre_rec(idx)), title('Precision + Recall')
% subplot(142)
% plot(stats.pre_rec_area(idx)), title('Precision/Recall Area')
subplot(132)
plot(stats.mean_precision(idx)), title('Precision')
subplot(133)
plot(stats.mean_recall(idx)), title('Recall')

% Visualize sorted alphas
figure;
for i = 1:size(alphas,1)
  subplot(2,size(alphas,1)/2,i)
  plot(alphas(i, idx)), title(['Alpha', num2str(i)])
end
 
% Visualize curve top-1 in the testset
top1 = idx(1);
str_top1 = sprintf('%05d', top1-1); % python notation
% load common plot definitions
plot_defs;

% parameters
params.exp_dir = results_dir;
params.dataset_name = 'PASCAL2007test';
params.prefix_output_files = strcat('results_',params.dataset_name);
params.save_output_files = 1;
params.set_log_scale = 1;
params.mean_precision = 1;
params.MATLAB = MATLAB;
params.PASCAL2007_test = 1;

params.exps = {{'exp30_07stats_NMS_05','STL, TopC=5'}, ...
               {'exp30_14stats_NMS_05','STL tuned, topC=5'}, ...
               {'exp14_04stats', 'SelectiveSearch, our exp'}, ...
               {'exp29_01stats', 'BING, our exp'},...
               };

visualize_plot_and_save(params);

