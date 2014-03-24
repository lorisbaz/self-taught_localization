function [bboxes_all, priority_all, img_width_all, img_height_all] = selective_search(testIms, outFiles, ss_version)
% Extract the SelectiveSearch bboxes, using the original pipeline
% described in the demo "demoPascal2007.m SelectiveSearch toolbox"
%
% If outFiles is not empty, then for each testIms{i} the function writes the MAT file outFiles{i}
% containing the variables:
%  img_width: width of the image
%  img_height: height of the image
%  bboxes: [N x 4] matrix, each row is [xmin, ymin, xmax, ymax]
%  priority: [N x 1] vector, each element containing the confidence value
%
% Example usage:
%  selective_search({'ILSVRC2012_val_00000001_n01751748.JPEG','ILSVRC2012_val_00000001_n01751748.JPEG'}, {'temp.mat', 'temp2.mat'})
%
% INPUT:
%  testIms: cell-array containing the list of the M image filenames
%  ss_version: either 'quality' (default), or 'fast'
%
% OUTPUT:
%  img_width_all: cell array of M elements, each being of type 'img_width'
%  img_height_all: cell array of M elements, each being of type 'img_height'
%  bboxes_all: cell array of M elements, each being of type 'bboxes'
%  priority_all: cell array of M elements, each being of type 'priority'

% input checks
if ~isempty(outFiles)
  assert(numel(testIms) == numel(outFiles));
end

% default input arguments
if nargin < 3
  ss_version = 'quality';
end
  
% add the path to the SelectiveSearch toolbox
addpath('/home/anthill/vlg/SelectiveSearchCodeIJCV');
addpath('/home/anthill/vlg/SelectiveSearchCodeIJCV/Dependencies');

% select the color spaces
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, ...
                      @SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
sigma = 0.8;

% After segmentation, filter out boxes which have a width/height smaller
% than minBoxWidth (default = 20 pixels).
minBoxWidth = 20;

% SelectiveSearch version
if strcmp(ss_version, 'fast')
  colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
  simFunctionHandles = simFunctionHandles(1:2); % Two different merging strategies
  ks = ks(1:2);
else
  assert(strcmp(ss_version, 'quality'));
end

% Test the boxes
totalTime = 0;
bboxes_all = {};
priority_all = {};
img_width_all = {};
img_height_all = {};
for i=1:length(testIms)
    fprintf('extracting SS for %s\n', testIms{i});
    try
      % reset the pseudo-random number generator, so to have deterministic results
      rng('default');
      % load the image
      im = imread(testIms{i});
      % try many Segmentation thresholds k
      idx = 1;
      for j=1:length(ks)
          k = ks(j); 
          minSize = k; % We set minSize = k
          % try many color spaces
          for n = 1:length(colorTypes)
              colorType = colorTypes{n};
              fprintf('k=%d, colorType=%s\n', k, colorType);
              tic;
              [boxesT{idx} blobIndIm blobBoxes hierarchy priorityT{idx}] = ...
                    Image2HierarchicalGrouping(im, sigma, k, minSize, colorType, simFunctionHandles);
              totalTime = totalTime + toc;
              idx = idx + 1;
          end
      end
      boxes = cat(1, boxesT{:}); % Concatenate boxes from all hierarchies
      priority = cat(1, priorityT{:}); % Concatenate priorities

      % Do pseudo random sorting as in paper
      priority = priority .* rand(size(priority));
      [priority sortIds] = sort(priority, 'ascend');
      boxes = boxes(sortIds,:);

      % filtering the bboxes by width (originally: call to FilterBoxesWidth)
      %boxes = FilterBoxesWidth(boxes, minBoxWidth);      
      [nr nc] = BoxSize(boxes);
      idsGood = (nr >= minBoxWidth) & (nc >= minBoxWidth);
      boxes = boxes(idsGood,:);
      priority = priority(idsGood);
      
      % remove the duplicates (originally: call to BoxRemoveDuplicates)      
      %boxes = BoxRemoveDuplicates(boxes);      
      [dummy uniqueIdx] = unique(boxes, 'rows', 'first');
      uniqueIdx = sort(uniqueIdx);
      boxes = boxes(uniqueIdx,:);
      priority = priority(uniqueIdx);

      % convert the boxes coordinates to the PASCAL standard [xmin, ymin, xmax, ymax]
      bboxes = [boxes(:,2) , boxes(:,1) , boxes(:,4) , boxes(:,3)];

      % image size
      img_width = size(im, 2);
      img_height = size(im, 1);
      
      % saving (if requested)
      if ~isempty(outFiles)
        save(outFiles{i}, 'bboxes', 'priority', 'img_width', 'img_height');
      end
      
      % return values
      bboxes_all{i} = bboxes;
      priority_all{i} = priority;
      img_width_all{i} = img_width;
      img_height_all{i} = img_height;  
    catch Exception
      Exception
      fprintf('AN ERROR HAS OCCURED!\n');      
    end
end
fprintf('\n');

fprintf('Time per image: %.2f\n', totalTime ./ length(testIms));

end

