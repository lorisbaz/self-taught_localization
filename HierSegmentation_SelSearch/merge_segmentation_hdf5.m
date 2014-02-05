function success = merge_segmentation_hdf5(dSET_)
% This function merges the segmentation results saved by the function 
% Segmentation_ImageNet in a single HDF5 database.
% This create a database in savePath (see Configuration file)
%
% Output:
% - returns 1 if everything worked, 0 otherwise

tiny_example = 0; % for debugging

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

% Configure the paths, the addpaths, etc.
Configure;

% Check the output directories
if ~exist(savePath, 'file')
    error('Results not found. Use Segmentation_ImageNet to compute them.');
end
if ~exist([savePath '/val'], 'file')
    error('Results not found. Use Segmentation_ImageNet to compute them.');
end
if ~exist([savePath '/train'], 'file')
    error('Results not found. Use Segmentation_ImageNet to compute them.');
end
if ~exist([savePath '/test'], 'file')
    error('Results not found. Use Segmentation_ImageNet to compute them.');
end

% Load image set
load_image_list;

% Set output file
database_file = [savePath 'segmetation_SS_' dSET_ '.h5'];

% Sequentially scan the results to create the database HDF5
success = create_hdf5_from_SS_segm(database_file, imageListClass, savePath);

