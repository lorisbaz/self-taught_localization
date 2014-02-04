% This demo shows how to use the software described in our IJCV paper: 
%   Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
% ---Modified by Loris Bazzani 01/2014 to run on the cluster and ImageNet
clear; clc;

% dataset selection
dSET_ = 'val'; % 'val', 'train' -> TODO: add training set
tiny_example = 1;

% Parameters
seg_params.colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
seg_params.simFunctionHandles = {@SSSimColourTextureSizeFillOrig, ...
                      @SSSimTextureSizeFill,@SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
seg_params.ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
seg_params.sigma = 0.8;

% Comment the following three lines for the 'quality' version
seg_params.colorTypes = seg_params.colorTypes(1:2); % 'Fast' uses HSV and Lab
seg_params.simFunctionHandles = seg_params.simFunctionHandles(1); % only the one of the paper
seg_params.ks = seg_params.ks(1:2);


% Configurations
rootPath = pwd;
toolboxPath = '/home/anthill/aleb/clients/aleb/aleb/toolbox/';
hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
switch (hostname)
    case 'anthill.cs.dartmouth.edu'
        imagePath = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012/';
        savePath  = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/segment_ILSVRC2012/';
        selectivePath = '/home/anthill/vlg/SelectiveSearchCodeIJCV/';        
        addpath(toolboxPath)
        
        run_on_anthill = 1;
        
    case 'alessandro-Linux'
        imagePath = '/home/alessandro/Data/ILSVRC2012';
        savePath  = 'TODO';
        selectivePath = 'TODO';
        
        run_on_anthill = 0;
        
    case 'lbazzani-desk'
        imagePath = '/home/lbazzani/DATASETS/ILSVRC2012/';
        savePath  = '/home/lbazzani/CODE/DATA/ILSVRC2012/segmentation/';
        selectivePath = '/home/lbazzani/CODE/3rd_part_libs/SelectiveSearchCodeIJCV/';

        run_on_anthill = 0;

end
% add required libraires
addpath([rootPath '/utils']);
addpath(genpath(selectivePath));
        
% create the output directories
if ~exist(savePath, 'file')
  mkdir(savePath);
end
if ~exist([savePath '/val'], 'file')
  mkdir([savePath '/val']);
end
if ~exist([savePath '/train'], 'file')
  mkdir([savePath '/train']);
end
if ~exist([savePath '/test'], 'file')
  mkdir([savePath '/test']);
end

if run_on_anthill
    % Anthill processes
    anthill = parallel.importProfile('/share/apps/matlab/anthill.settings');
    parallel.defaultClusterProfile(anthill);
end


% Check if Compiled 
% ...anisotropic gaussian filter
if(~exist('anigauss'))
    fprintf('Compile the anisotropic gauss filtering with \n');
    fprintf([' mex Dependencies/anigaussm/anigauss_mex.c Dependencies/'...
              'anigaussm/anigauss.c -output anigauss'])
end
% ...word counter
if(~exist('mexCountWordsIndex'))
    fprintf('Compile the word counter with \n');
    fprintf(' mex Dependencies/mexCountWordsIndex.cpp');
end
% ...the code of Felzenszwalb and Huttenlocher, IJCV 2004.
if(~exist('mexFelzenSegmentIndex'))
    fprintf('Compiling the segmentation algorithm with\n');
    fprintf([' mex Dependencies/FelzenSegment/mexFelzenSegmentIndex.cpp'...
             ' -output mexFelzenSegmentIndex;'])
end


% Load image set
switch dSET_
    case 'val'
        % Load image list
        imageList_filename = [imagePath 'val_images.txt'];
        filestream = fopen(imageList_filename);
        imageList = textscan(filestream,'%s','delimiter','\n'); % read image list
        imageList = imageList{1};
        fclose(filestream);
        
        % Load labels list 
        labelList_filename = [imagePath 'val_labels.txt'];
        filestream = fopen(labelList_filename);
        labelList = textscan(filestream,'%d','delimiter','\n'); % read image list
        labelList = labelList{1};
        fclose(filestream);
        
        % organize tasks by class (like in the python code)
        n_classes = max(labelList);
        if tiny_example
            n_classes = 2;
        end
        imageListClass = cell(1,n_classes);
        for i = 1:n_classes
            idx = find(labelList==i);
            if tiny_example
                idx = idx(1:5);
            end
            imageListClass{i} = [imageList(idx)];
        end
    
    %case 'test'    
    %case 'train'  
    otherwise
        error('Not implemented yet. Only Validation is supported.')
end

% Segmentation - Run
if run_on_anthill  % ...on the cluster
    cluster_opts.qsubargs = '-l ironfs -l h_rt=00:10:00 -l virtual_free=2G -l mem_free=2G';
    cluster_opts.paths = {rootPath, [rootPath '/utils'], toolboxPath, genpath(selectivePath)};
    num_tasks = n_classes;  % Note: set to 0 for debugging
    % Run tasks on cluster
    val = parallelize_function(@extract_segmentation, {imageListClass, ...
                               imagePath, savePath, seg_params}, ...
                               [1, 0, 0, 0], num_tasks, cluster_opts);
    
else      % ...on standard PC
    extract_segmentation(imageList, imagePath, savePath, seg_params);
end
fprintf('\n');