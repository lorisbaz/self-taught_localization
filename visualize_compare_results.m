% This script load a set of results computer by the exp08_* python scripts
% and visualize the curves for comparison
clear;

% set some colors
MATLAB.Color.yellow = [1 1 0];
MATLAB.Color.magenta = [1 0 1];
MATLAB.Color.orange = [0 1 1];
MATLAB.Color.cyan = [0 1 1]; % cyan
MATLAB.Color.cyanMed = [0 0.6 0.6]; % cyanMed
MATLAB.Color.cyanDark = [0 0.3 0.3]; % cyanDark
MATLAB.Color.red = [1 0 0];
MATLAB.Color.redMed = [0.6 0 0];
MATLAB.Color.redDark = [0.3 0 0];
MATLAB.Color.green = [0 1 0];
MATLAB.Color.blue = [0 0 1];
MATLAB.Color.white = [1 1 1];
MATLAB.Color.black = [0 0 0];
MATLAB.Color.orange = [255 150 0]./255;
MATLAB.Color.orangeLight = [255 170 0]./255;
MATLAB.Color.orangeMed = [255 140 0]./255;
MATLAB.Color.orangeDark = [255 110 0]./255;
MATLAB.Color.greenMed = [45 125 45]./255;
MATLAB.Color.greenDark = [22 62 22]./255; % green medium
MATLAB.Color.greyLight = [90 90 90]./255;
MATLAB.Color.greyMed = [180 180 180]./255;
MATLAB.Color.greyDark = [230 230 230]./255;
MATLAB.Color.orange2 = [225 122 0]./255; % orange
MATLAB.Color.brown = [128 64 0]./255; % brown
MATLAB.Color.brownLight = [255 128 0]./255; % brown light
MATLAB.Color.brownDark = [64 32 0]./255; % brown dark
MATLAB.Colors_all = {MATLAB.Color.yellow, MATLAB.Color.magenta, MATLAB.Color.orange, MATLAB.Color.cyan, MATLAB.Color.cyanMed, MATLAB.Color.cyanDark, MATLAB.Color.red, MATLAB.Color.redMed, MATLAB.Color.redDark, MATLAB.Color.green , MATLAB.Color.blue, MATLAB.Color.white, MATLAB.Color.black, MATLAB.Color.orange, MATLAB.Color.greenMed, MATLAB.Color.greenDark, MATLAB.Color.greyLight, MATLAB.Color.greyMed, MATLAB.Color.greyDark, MATLAB.Color.orange2, MATLAB.Color.brown};


% init
i = 0;
exp_set = struct([]);
experiments_output_directory = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
plot_GT = 0;
plot_Top5 = 1;
plot_SS = 1;
plot_BING_Pascal2007 = 0;

% *** GT
if plot_GT
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
if plot_Top5
%     i = i + 1;
%     exp_set(i).name = 'exp08_08';
%     exp_set(i).spec = 'Top 5 labels';
%     exp_set(i).method = 'GraySegm';
%     exp_set(i).color = [0.4177    0.6981    0.0326];
    
    i = i + 1;
    exp_set(i).name = 'exp08_09';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'GrayBox';
    exp_set(i).color = MATLAB.Color.orange;
        
    i = i + 1;
    exp_set(i).name = 'exp06_12';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'GraySegm';
    exp_set(i).color = MATLAB.Color.brown;
    
    i = i + 1;
    exp_set(i).name = 'exp08_10';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'ClassicSliding';
    exp_set(i).color = MATLAB.Color.cyan;
    
%     i = i + 1;
%     exp_set(i).name = 'exp08_13';
%     exp_set(i).spec = 'Top 5 labels';
%     exp_set(i).method = 'Sliding over GraySegm heatmaps';
%     exp_set(i).color = MATLAB.Color.black;
    
    i = i + 1;
    exp_set(i).name = 'exp08_16';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'GraySegm+GrayBox';
    exp_set(i).color = MATLAB.Color.blue;
 
    i = i + 1;
    exp_set(i).name = 'exp22_01';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'Re-rank GrayBox';
    exp_set(i).color = MATLAB.Color.red;
    
    i = i + 1;
    exp_set(i).name = 'exp22_02';
    exp_set(i).spec = 'Top 5 labels';
    exp_set(i).method = 'Re-rank GraySegm+GrayBox';
    exp_set(i).color = MATLAB.Color.redDark;
end

% *** SS
if plot_SS
    i = i + 1;
    exp_set(i).name = 'exp08_12';
    exp_set(i).spec = '';
    exp_set(i).method = 'SS (fast)';
    exp_set(i).color = MATLAB.Color.magenta;
end

figure(121);
hold on;
legend_stuff = cell(length(exp_set),1);
for k = 1:length(exp_set)
    try
        load([experiments_output_directory '/' exp_set(k).name 'stats'...
        '/mat/recall_vs_numPredBboxesImage.mat'])
    catch
        load([experiments_output_directory '/' exp_set(k).name ...
        '/mat/recall_vs_numPredBboxesImage.mat'])
    end
    try
        plot(eval(['x_values_' exp_set(k).name]), eval(['recall_' ...
            exp_set(k).name]),'Color',exp_set(k).color, 'LineWidth', 3)
    catch
        plot(x_values, recall, 'Color',exp_set(k).color, 'LineWidth', 3)
    end
    legend_stuff{k} =  [exp_set(k).spec ' - ' exp_set(k).method ' (' ...
        exp_set(k).name ')'];
end

% *** BING
if plot_BING_Pascal2007
    load('plot_defs.mat');
    i = i + 1;
    exp_set(i).name = '(from paper)';
    exp_set(i).spec = 'BING';
    exp_set(i).method = 'DR (Pascal 2007)';
    exp_set(i).color = MATLAB.Color.red;
    plot(1:numel(DR), DR, 'Color', exp_set(i).color, 'LineWidth', 3);
    legend_stuff{i} =  [exp_set(i).spec ' - ' exp_set(i).method ' (' ...
        exp_set(i).name ')'];    
end

hold off;
grid on;
legend(legend_stuff, 'Interpreter', 'none')
axis([1, 50, 0, 1])
xlabel('Num subwindows')
ylabel('Recall')
title('Results on ILSVRC2012-val-200')



