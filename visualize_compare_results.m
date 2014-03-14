% This script load a set of results computer by the exp08_* python scripts
% and visualize the curves for comparison
clear;

% init
i = 0;
experiments_output_directory = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';

% GT
% i = i + 1;
% exp_set(i).name = 'exp08_03';
% exp_set(i).spec = 'GT label'; 
% exp_set(i).method = 'GrayBox';

i = i + 1;
exp_set(i).name = 'exp08_05';
exp_set(i).spec = 'GT label';
exp_set(i).method = 'ClassicSliding';

i = i + 1;
exp_set(i).name = 'exp08_07';
exp_set(i).spec = 'GT label';
exp_set(i).method = 'GraySegm';

% TopC
i = i + 1;
exp_set(i).name = 'exp08_08';
exp_set(i).spec = 'Top 5 labels'; 
exp_set(i).method = 'GraySegm';

i = i + 1;
exp_set(i).name = 'exp08_09';
exp_set(i).spec = 'Top 5 labels';
exp_set(i).method = 'GrayBox';

i = i + 1;
exp_set(i).name = 'exp08_10';
exp_set(i).spec = 'Top 5 labels'; 
exp_set(i).method = 'ClassicSliding';

figure(121);
hold on;
colors_rnd = rand(length(exp_set),3);
legend_stuff = cell(length(exp_set),1);
for i = 1:length(exp_set)
    load([experiments_output_directory '/' exp_set(i).name ...
                        '/mat/recall_vs_numPredBboxesImage.mat'])
    plot(eval(['x_values_' exp_set(i).name]), eval(['recall_' ...
                exp_set(i).name]),'Color',colors_rnd(i,:), 'LineWidth', 3)
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

