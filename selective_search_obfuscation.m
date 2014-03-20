function selective_search_onfuscation(testIms, outFiles, ss_version)
% Extract the SelectiveSearch bboxes and masks that will be used for
% obfuscation
%
% Each MAT output file will contain the following variables:
%  img_width: width of the image
%  img_height: height of the image
%  hBlobs: structure that contains 'mask', 'rect', 'size'
%
% Example usage:
%  selective_search_obfuscation({'ILSVRC2012_val_00000001_n01751748.JPEG',
%        'ILSVRC2012_val_00000001_n01751748.JPEG'}, {'temp.mat', 'temp2.mat'})
%
% INPUT:
%  testIms: cell-array containing the list of image filenames
%  ss_version: either 'quality' (default), or 'fast'
%
% OUTPUT:
%  nothing, the function writes the output to files.

assert(numel(testIms) == numel(outFiles));
if nargin < 3
  ss_version = 'quality';
end
  
% add the path to the SelectiveSearch toolbox
addpath('/home/anthill/vlg/SelectiveSearchCodeIJCV');
addpath('/home/anthill/vlg/SelectiveSearchCodeIJCV/Dependencies');
addpath('./HierSegmentation_SelSearch/utils')

% select the color spaces
colorTypes = {'Hsv', 'Lab', 'RGI', 'H', 'Intensity'};

% Here you specify which similarity functions to use in merging
simFunctionHandles = {@SSSimColourTextureSizeFillOrig, @SSSimTextureSizeFill, ...
                      @SSSimBoxFillOrig, @SSSimSize};

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
sigma = 0.8;

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
for i=1:length(testIms)
    fprintf('extracting SS for %s\n', testIms{i});
    try
      % reset the pseudo-random number generator, so to have deterministic results
      rng('default');
      % load the image
      im = imread(testIms{i});
      % try many Segmentation thresholds k
      idx = 1; idx2 = 1;
      for j=1:length(ks)
          k = ks(j); 
          minSize = k; % We set minSize = k
          % try many color spaces
          for n = 1:length(colorTypes)
              colorType = colorTypes{n};
              fprintf('k=%d, colorType=%s\n', k, colorType);
              tic;
              [boxesT{idx} blobIndIm{idx} blobBoxes hierarchy priorityT{idx}] = ...
                    Image2HierarchicalGrouping(im, sigma, k, minSize, colorType,...
                    simFunctionHandles);
              totalTime = totalTime + toc;
              
              % Get blobs of initial segmentation
              segmentation = SegmentIndices2Blobs(blobIndIm{idx}, blobBoxes);
              for h = 1:length(hierarchy)
                  % recreate the tree to be saved
                  tree{idx2} = ...
                      RecreateBlobHierarchyLevelsTree(segmentation, ...
                      hierarchy{h});
                  
                  hBlobs{idx2} = RecreateBlobHierarchyIndIm(blobIndIm{idx}, ...
                      blobBoxes, hierarchy{h});
                  
                  idx2 = idx2 + 1;
              end
              
              idx = idx + 1;
          end
      end

      % saving
      img_width = size(im, 2);
      img_height = size(im, 1);
      save(outFiles{i}, 'blobIndIm', 'tree','hBlobs', 'img_width', 'img_height');
    catch Exception
      Exception
      fprintf('AN ERROR HAS OCCURED!\n');      
    end
end
fprintf('\n');

fprintf('Time per image: %.2f\n', totalTime ./ length(testIms));

end

