function selective_search_obfuscation_optimized(testIms, outFiles, ss_version)
% Extract the Felzenswalb bboxes and masks used by selective search
%
% Each MAT output file will contain the following variables:
%  img_width: width of the image
%  img_height: height of the image
%  blobIndIm: structure that contains 'mask', 'rect', 'size'
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

% Thresholds for the Felzenszwalb and Huttenlocher segmentation algorithm.
% Note that by default, we set minSize = k, and sigma = 0.8.
ks = [50 100 150 300]; % controls size of segments of initial segmentation. 
sigma = 0.8;

% SelectiveSearch version
if strcmp(ss_version, 'fast')
  colorTypes = colorTypes(1:2); % 'Fast' uses HSV and Lab
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
      % if the image is in grayscale, we convert it to colors
      if size(im, 3) == 1
        im = repmat(im, [1, 1, 3]);
      end      
      % try many Segmentation thresholds k
      idx = 1;
      blobIndIm = cell(1, length(ks)*length(colorTypes));
      for j=1:length(ks)
          k = ks(j); 
          minSize = k; % We set minSize = k
          % try many color spaces
          for n = 1:length(colorTypes)
              colorType = colorTypes{n};
              fprintf('k=%d, colorType=%s\n', k, colorType);
              tic;
              [colourIm, imageToSegment] = Image2ColourSpace(im, colorType);
              [blobIndIm{idx}, blobBoxes, neighbours] = mexFelzenSegmentIndex(imageToSegment, sigma, k, minSize);
              totalTime = totalTime + toc;
              
              idx = idx + 1;
          end
      end

      % saving
      img_width = size(im, 2);
      img_height = size(im, 1);
      save(outFiles{i}, 'blobIndIm', 'img_width', 'img_height');
    catch Exception
      Exception
      fprintf('AN ERROR HAS OCCURED!\n');      
    end
end
fprintf('\n');

fprintf('Time per image: %.2f\n', totalTime ./ length(testIms));

end

