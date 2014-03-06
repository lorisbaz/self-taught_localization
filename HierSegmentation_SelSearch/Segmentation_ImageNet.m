function Segmentation_ImageNet(dSET_)
% Run segmentation of [1] on ImageNet
%    dSET_ = 'val or 'train'
%   [1] Selective Search for Object Recognition,
%   J.R.R. Uijlings, K.E.A. van de Sande, T. Gevers, A.W.M. Smeulders, IJCV 2013
% ---Modified by Loris Bazzani 01/2014 to run on the cluster and ImageNet

tiny_example = 0; % for debugging

% Parameters
fix_sz = 600; % resize images that are bigger that resize_big_images
                          % set it to 0 if you want to disable it
                          
seg_params.colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};
seg_params.central_crop = 0;
seg_params.SelSearchExp = 1;


% Here you specify which similarity functions to use in merging
seg_params.simFunctionHandles = {@SSSimColourTextureSizeFillOrig, ...
                      @SSSimTextureSizeFill,@SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
seg_params.ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
seg_params.sigma = 0.8;

% Comment the following three lines for the 'quality' version
seg_params.colorTypes = seg_params.colorTypes(1:2); % 'Fast' uses HSV and Lab
seg_params.simFunctionHandles = seg_params.simFunctionHandles(1:2); % only the one of the paper
seg_params.ks = seg_params.ks(1:2);

fprintf('---Parameters setup---\n')
fprintf([' * k: ', num2str(seg_params.ks), '\n'])
fprintf(' * min_segm: same as k \n')
fprintf([' * sigma: ', num2str(seg_params.sigma), '\n'])

% Configure the paths, the addpaths, etc.
Configure;
    
% create the output directories
if ~exist(savePath, 'file')
  mkdir(savePath);
end
%run_on_anthill = 0;

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
load_image_list;

% Segmentation - Run
fprintf('---Starting processing---\n')
if run_on_anthill  % ...on the cluster
    cluster_opts.qsubargs = '-l ironfs -l h_rt=02:59:00 -l virtual_free=1G -l mem_free=1G';
    cluster_opts.paths = {rootPath, [rootPath '/utils'], toolboxPath, genpath(selectivePath)};
    num_tasks = 100;  % Note: set to 0 for debugging
    % Run tasks on cluster
    val = parallelize_function(@extract_segmentation, {imageListClass, ...
                               imagePath, savePath, seg_params, fix_sz}, ...
                               [1, 0, 0, 0, 0], num_tasks, cluster_opts);
    
else      % ...on standard PC
    for i = 1:n_classes 
        extract_segmentation(imageListClass{i}, imagePath, savePath, seg_params, fix_sz);
    end
end
fprintf('\n');
