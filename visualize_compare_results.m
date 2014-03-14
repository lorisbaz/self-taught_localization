% This script load a set of results computer by the exp08_* python scripts
% and visualize the curves for comparison
clear;

% init
i = 0;
experiments_output_directory = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
no_GT = 0;
no_Top5 = 1;

% *** GT
if ~no_GT
    i = i + 1;
    exp_set(i).name = 'exp08_03';
    exp_set(i).spec = 'GT label';
    exp_set(i).method = 'GrayBox';
    exp_set(i).color = [0.4228    0.7011    0.1280];
    
    i = i + 1;
    exp_set(i).name = 'exp08_05';
    exp_set(i).spec = 'GT label';
    exp_set(i).method = 'ClassicSliding';
    exp_set(i).color = [0.5479    0.6663    0.9991];
    
    i = i + 1;
    exp_set(i).name = 'exp08_07';
    exp_set(i).spec = 'GT label';
    exp_set(i).method = 'GraySegm';
    exp_set(i).color = [0.9427    0.5391    0.1711];
end

% *** Top 5
if ~no_Top5
    i = i + 1;
    exp_set(i).name = 'exp08_08';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'GraySegm';
    exp_set(i).color = [0.4177    0.6981    0.0326];
    
    i = i + 1;
    exp_set(i).name = 'exp08_09';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'GrayBox';
    exp_set(i).color = [0.9831    0.6665    0.5612];
    
    i = i + 1;
    exp_set(i).name = 'exp08_10';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'ClassicSliding';
    exp_set(i).color = [0.3015    0.1781    0.8819];
end

figure(121);
hold on;
legend_stuff = cell(length(exp_set),1);
for i = 1:length(exp_set)
    load([experiments_output_directory '/' exp_set(i).name ...
        '/mat/recall_vs_numPredBboxesImage.mat'])
    plot(eval(['x_values_' exp_set(i).name]), eval(['recall_' ...
        exp_set(i).name]),'Color',exp_set(i).color, 'LineWidth', 3)
    legend_stuff{i} =  [exp_set(i).spec ' - ' exp_set(i).method ' (' ...
        exp_set(i).name ')'];
end
hold off;
grid on;
legend(legend_stuff)
axis([1, 15, 0, 1])
xlabel('Num subwindows')
ylabel('Recall')
title('Results on ImageNet(200)')

