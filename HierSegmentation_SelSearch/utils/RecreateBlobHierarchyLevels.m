function [real_hierarchy, hBlobs] = RecreateBlobHierarchyLevels(img_segm, blobs, hierarchy)
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

n_segm_Li = length(blobs);
maxid_segm_Li = n_segm_Li + 1;

real_hierarchy{1} = img_segm; % first level is just Felz. output
coverage = zeros(length(hierarchy) + 1,1);
coverage(1:length(blobs)) = 1;

L = 2; % set starting level
while (length(hBlobs)>maxid_segm_Li)
    real_hierarchy{L} = real_hierarchy{L-1}; % init accordingly to previous hierarchy
    
    n_new_segm_Li1 = 0;
    for i=length(blobs)+1:length(hBlobs)
        n = find(hierarchy == i);
        
        if n(1)<=maxid_segm_Li && n(2)<=maxid_segm_Li
            if length(n) ~= 2
                error('One can not merge more than 2 blobs!');
            end
            hBlobs{i} = MergeBlobs(hBlobs{n(1)}, hBlobs{n(2)});
            
            % overwrite the merged segment with the new label
%             TMP_img = real_hierarchy{L}(hBlobs{i}.rect(1):hBlobs{i}.rect(3),hBlobs{i}.rect(2):hBlobs{i}.rect(4));
%             TMP_img(hBlobs{i}.mask) = i;
%             real_hierarchy{L}(hBlobs{i}.rect(1):hBlobs{i}.rect(3),hBlobs{i}.rect(2):hBlobs{i}.rect(4)) = TMP_img;           
%             if sum(hBlobs{i}.mask(:))~=sum(sum(real_hierarchy{L}==n(1)))+sum(sum(real_hierarchy{L}==n(2)))
%                fprintf('fanculo')  % heavy debug!!! and back engineer!!! 
%             end
            real_hierarchy{L}(real_hierarchy{L}==n(1)) = i;
            real_hierarchy{L}(real_hierarchy{L}==n(2)) = i;

            n_new_segm_Li1 = n_new_segm_Li1 + 1;
            
            coverage(n(1)) = 1;
            coverage(n(2)) = 1;      
            %imagesc(real_hierarchy{L})
            
        end
        
    end
    maxid_segm_Li = n_segm_Li + n_new_segm_Li1;

    L = L + 1;
end
    