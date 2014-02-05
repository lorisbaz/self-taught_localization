% function merge_segmentation_hdf5(dSET_)
% This function merges the segmentation results saved by the function 
% Segmentation_ImageNet in a single HDF5 database.


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


% Configurations
rootPath = pwd;
toolboxPath = '/home/anthill/aleb/clients/aleb/aleb/toolbox/';
hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
switch (hostname)
    case 'anthill.cs.dartmouth.edu'
        imagePath = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012/';
        trainPath = [imagePath 'train/'];
        valPath = [imagePath 'val/'];
        testPath = [imagePath 'test/'];
        savePath  = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/segment_ILSVRC2012/';
        selectivePath = '/home/anthill/vlg/SelectiveSearchCodeIJCV/';        
        addpath(toolboxPath)
        
        run_on_anthill = 1;
        
    case 'alessandro-Linux'
        imagePath = '/home/alessandro/Data/ILSVRC2012/';
        trainPath = [imagePath 'train/'];
        valPath = [imagePath 'val/'];
        testPath = [imagePath 'test/'];
        savePath  = 'TODO';
        selectivePath = 'TODO';
        
        run_on_anthill = 0;
        
    case 'lbazzani-desk'
        imagePath = '/home/lbazzani/DATASETS/ILSVRC2012/';
        trainPath = [imagePath 'train/'];
        valPath = [imagePath 'val/'];
        testPath = [imagePath 'test/'];
        savePath  = '/home/lbazzani/CODE/DATA/ILSVRC2012/segmentation/';
        selectivePath = '/home/lbazzani/CODE/3rd_part_libs/SelectiveSearchCodeIJCV/';

        run_on_anthill = 0;

end


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
fprintf('---Prepare computation---\n')
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
               
    case 'train'
        classList_filename = [imagePath 'classid_wnid_words.txt'];
        filestream = fopen(classList_filename);
        imageList = textscan(filestream,'%d %s %s','delimiter','\t'); % read image list
        %imageList = imageList{1};
        fclose(filestream);
        
        labelList = imageList{1};
        labelList_name = imageList{2};
        
        % organize tasks by class (like in the python code)
        n_classes = max(labelList);
        if tiny_example
            n_classes = 2;
        end
        imageListClass = cell(1,n_classes);
        for i = 1:n_classes            
            imlist = dir([trainPath labelList_name{i} '/*.JPEG']);
            n_img = length(imlist);
            if tiny_example
                n_img = 5;
            end
            for j = 1:n_img       
                imageListClass{i}{j,1} = ['train/' labelList_name{i} '/' imlist(j).name];
            end
            if ~exist([savePath 'train/' labelList_name{i}], 'file')
                mkdir([savePath 'train/' labelList_name{i}]);
            end
            
        end
        
    case 'test'
        error('Not implemented yet.')
        
    otherwise
        error('Not implemented yet.')
end


for i = 1:n_classes
    
end

