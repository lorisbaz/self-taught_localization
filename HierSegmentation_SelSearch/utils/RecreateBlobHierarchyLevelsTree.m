function tree = RecreateBlobHierarchyLevelsTree(blobs, hierarchy)
% [blobs hierarchy] = RecreateBlobHierarchy(img_segm, blobs, hierarchy)
% 
% Recreates the hierarchical grouping using the starting blobs and the 
% resulting hierarchy. This allows one to save the grouping using
% relatively small disk space while still being able to fastly recreate the
% complete grouping.
%
% blobs:            Input cell array with blobs
% hierarchy:        Hierarchy of the blobs as created by
%                   HierarchicalGrouping.m
%
% hBlobs:           All segments of the hierarchical grouping.
%
%     Loris Bazzani - 2014

hBlobs = cell(length(hierarchy) + 1,1);

hBlobs(1:length(blobs)) = blobs;

tree.leaves = 1:length(blobs); % segments
tree.nodes = [];
for i=length(blobs)+1:length(hBlobs)
    n = find(hierarchy == i);
    
    tree.nodes = [tree.nodes; [i, n]];
end