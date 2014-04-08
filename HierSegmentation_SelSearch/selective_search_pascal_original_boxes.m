function selective_search_pascal_original_boxes()
% This script calculates the recall and MABO scores for the PASCAL 2007 dataset,
% using the original pipeline of the SS code.
% In particular, this script assumes that you have already ran the "demoPascal2007.m" 
% script under, /home/anthill/vlg/SelectiveSearchCodeIJCV, saving all the variables
% (e.g. the "boxes" in particular) to the file
% /home/anthill/vlg/SelectiveSearchCodeIJCV/demoPascal2007.mat

% output file
params.out_file = '/home/anthill/vlg/SelectiveSearchCodeIJCV/demoPascal2007.mat';
% gt file
params.gt_bboxes = ['/home/anthill/vlg/SelectiveSearchCodeIJCV/GroundTruthVOC2007test.mat'];
% max_num_pred_bboxes
params.max_num_pred_bboxes = [1:20, 25, 30, 35, 40, 50, 100, 200, 300, 400, 500, 750, 1000, 1500, 2500];


% load the pred and gt boxes
load(params.gt_bboxes);
load(params.out_file);


if 0
  mabo_all = [];
  for i=1:numel(params.max_num_pred_bboxes)
    % Get for each ground truth box the best Pascal Overlap Score
    max_num_pred_bboxes = params.max_num_pred_bboxes(i);
    % make a copy of bboxes
    boxes2 = {};
    for j=1:numel(boxes)
      N = size(boxes{j}, 1);
      boxes2{j} = boxes{j}(1:min(N,max_num_pred_bboxes), :);
    end
    % calculate the MABO
    [boxAbo boxMabo boScores avgNumBoxes] = BoxAverageBestOverlap(gtBoxes, gtImIds, boxes2);
    mabo_all(i) = boxMabo;
  end
  h=figure;
  hold on;
  xlabel('num bboxes per image');
  ylabel('MABO');
  plot(params.max_num_pred_bboxes, mabo_all, '-mx');
end

 
if 1
  recall_mean_all = [];
  for i=1:numel(params.max_num_pred_bboxes)
    fprintf('num_pred_bboxes: %d/%d\n', i, numel(params.max_num_pred_bboxes));
    % Get for each ground truth box the best Pascal Overlap Score
    max_num_pred_bboxes = params.max_num_pred_bboxes(i);
    % for each gt bbox, get the max overlap
    maxScores = MaxOverlapScores(gtBoxes, gtImIds, boxes, max_num_pred_bboxes);
    % Get recall per class
    for cI=1:length(maxScores)
        recall(cI) = sum(maxScores{cI} > 0.5) ./ length(maxScores{cI});
        averageBestOverlap(cI) = mean(maxScores{cI});
    end
    recall_mean_all(i) = mean(recall);
  end
  %save([params.out_dir '/temp_recall_mean.mat'], 'recall_mean_all');
  h=figure;
  hold on;
  xlabel('num bboxes per image');
  ylabel('mean recall (over the classes)');
  plot(params.max_num_pred_bboxes, recall_mean_all, '-mx');
  %saveas(h, [params.out_dir '/recall_mean.fig']);  
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
        N = size(testBoxes{testIds}, 1);
        bb = testBoxes{testIds}(1:min(N,max_num_pred_bboxes), :);
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


