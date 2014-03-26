function selective_search_pascal()
% This script is a debugging script that computes the SS performance on PASCAL VOC,
% using the same Matlab script we use in our Python wrapper.
%
% NOTE: this is very hacky code, not meant to be really used :)

addpath('~/toolbox');
addpath('/home/anthill/vlg/SelectiveSearchCodeIJCV');
addpath('/home/anthill/vlg/SelectiveSearchCodeIJCV/Dependencies');
addpath('/home/anthill/aleb/clients/vlg/vlg/experimental/grayobfuscation');

params = [];

% pascal 2007 root dir
params.root_dir = '/home/ironfs/scratch/vlg/Data/Images/PASCAL_VOC_2007';
% output directory
params.out_dir = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/test_aleb_shuffle';
% ss version
params.ss_version = 'fast';
% output file
params.out_file = [params.out_dir '/out.mat'];
% gt file
params.gt_bboxes = ['/home/anthill/vlg/SelectiveSearchCodeIJCV/GroundTruthVOC2007test.mat'];
% max_num_pred_bboxes
params.max_num_pred_bboxes = [50, 100, 500, 1000, 2500];

% create out directories
if ~exist(params.out_dir, 'file')
  mkdir(params.out_dir);
  mkdir([params.out_dir '/testImgs']);
end
  
if ~exist(params.out_file, 'file')
  % read the list of test images
  fd = fopen([params.root_dir '/ImageSets/Main/test.txt']);
  testImgs = textscan(fd, '%s');
  fclose(fd);
  testImgs = testImgs{1};

  % extract the SS subwindows
  options3 = [];
  options3.memory_req = 2;
  options3.time_req = 2;
  fun_handler = @selective_search_pascal_support;
  fun_parameters = {testImgs, params};
  fun_parameters_to_parallize = [1 0];
  num_tasks = 500;
  %parallelize_function(fun_handler, fun_parameters, fun_parameters_to_parallize, num_tasks, options3);

  % compact everything in a single file
  load(params.gt_bboxes, 'testIms');

  %  img_width: width of the image
  %  img_height: height of the image
  %  bboxes: [N x 4] matrix, each row is [xmin, ymin, xmax, ymax]
  %  priority: [N x 1] vector, each element containing the confidence value
  pred_bboxes = {};
  for i=1:numel(testIms)
     fprintf('%d / %d\n', i, numel(testIms));
     outfile = [params.out_dir '/testImgs/' testIms{i} '.mat'];
     S = load(outfile);
     pred_bboxes{i} = [];
     pred_bboxes{i}.img_width = S.img_width;
     pred_bboxes{i}.img_height = S.img_height;
     pred_bboxes{i}.bboxes = S.bboxes;
     pred_bboxes{i}.confidence = S.confidence;
  end
  
  % saving
  save(params.out_file, 'pred_bboxes', '-v7.3');
end

% load the pred and gt boxes
load(params.gt_bboxes);
load(params.out_file);

% sort the confidence and swap x/y
for i=1:numel(pred_bboxes)
  aa = pred_bboxes{i}.confidence;
  [~, idxsort] = sort(aa, 'descend');
  pred_bboxes{i}.confidence = pred_bboxes{i}.confidence(idxsort);
  bbb = pred_bboxes{i}.bboxes(idxsort,:);
  bbb = [bbb(:,2), bbb(:,1), bbb(:,4), bbb(:,3)];
  pred_bboxes{i}.bboxes = bbb;
end


if 1
  recall_all = [];
  for i=1:numel(params.max_num_pred_bboxes)
    fprintf('%d / %d\n', i, numel(params.max_num_pred_bboxes));
    % Get for each ground truth box the best Pascal Overlap Score
    max_num_pred_bboxes = params.max_num_pred_bboxes(i);
    maxScores = MaxOverlapScores(gtBoxes, gtImIds, pred_bboxes, max_num_pred_bboxes);
    maxScoresFlat = horzcat(maxScores{:});
    recall = sum(maxScoresFlat > 0.5) / numel(maxScoresFlat);
    recall_all(i) = recall;
  end
  save([params.out_dir '/temp_recall.mat'], 'recall_all');
  h=figure;
  hold on;
  xlabel('num bboxes per image');
  ylabel('recall');
  plot(params.max_num_pred_bboxes, recall_all, '-bo');
  saveas(h, [params.out_dir '/recall.fig']);  
end
  
if 1
  recall_mean_all = [];
  for i=1:numel(params.max_num_pred_bboxes)
    % Get for each ground truth box the best Pascal Overlap Score
    max_num_pred_bboxes = params.max_num_pred_bboxes(i);
    maxScores = MaxOverlapScores(gtBoxes, gtImIds, pred_bboxes, max_num_pred_bboxes);
    % Get recall per class
    for cI=1:length(maxScores)
        recall(cI) = sum(maxScores{cI} > 0.5) ./ length(maxScores{cI});
        averageBestOverlap(cI) = mean(maxScores{cI});
    end
    recall_mean_all(i) = mean(recall);
  end
  save([params.out_dir '/temp_recall_mean.mat'], 'recall_mean_all');
  h=figure;
  hold on;
  xlabel('num bboxes per image');
  ylabel('mean recall (over the classes)');
  plot(params.max_num_pred_bboxes, recall_mean_all, '-rx');
  saveas(h, [params.out_dir '/recall_mean.fig']);  
end

end


function scores = MaxOverlapScores(gtBoxes, gtImIds, testBoxes, max_num_pred_bboxes)
% Get best overlap scores for each box in gtBoxes
% gtImIds contains for each ground truth box the index of the corresponding
% image in which it can be found. This index again corresponds
% with the testBoxes cell-array
for cI = 1:length(gtBoxes) % For all classes
    classBoxes = gtBoxes{cI}; 
    for i=1:length(classBoxes)
        % Get a single GT box        
        testIds = gtImIds{cI}(i);
        % Calculate Pascal Overlap score and take best
        N = size(testBoxes{testIds}.bboxes, 1);
        bb = testBoxes{testIds}.bboxes(1:min(N,max_num_pred_bboxes), :);
        scores{cI}(i) = max(OverlapScores(classBoxes(i,:), bb));
    end
end
end


function scores = OverlapScores(gtBox, testBoxes)
% Get pascal overlap scores. gtBox versus all testBoxes


gtBoxes = repmat(gtBox, size(testBoxes,1), 1);
intersectBoxes = BoxIntersection(gtBoxes, testBoxes);
overlapI = intersectBoxes(:,1) ~= -1; % Get which boxes overlap

% Intersection size
[nr nc intersectionSize] = BoxSize(intersectBoxes(overlapI,:));

% Union size
[nr nc testBoxSize] = BoxSize(testBoxes(overlapI,:));
[nr nc gtBoxSize] = BoxSize(gtBox);
unionSize = testBoxSize + gtBoxSize - intersectionSize;

scores = zeros(size(testBoxes,1),1);
scores(overlapI) = intersectionSize ./ unionSize;
end


function val = selective_search_pascal_support(testImg, params)
val = 0;

system('uname -a');
addpath('/home/anthill/aleb/clients/vlg/vlg/experimental/grayobfuscation');

infile = [params.root_dir '/JPEGImages/' testImg '.jpg'];
outfile = [params.out_dir '/testImgs/' testImg '.mat'];
selective_search({infile}, {outfile}, params.ss_version);

val = 1;
end

