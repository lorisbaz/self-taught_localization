function val = extract_segmentation(testIms, imagePath, savePath, seg_params)

% output value
val = 0;

totalTime = 0;
for i=1:length(testIms)
    fprintf('Elaborating %d / %d %s\n', i, length(testIms), testIms{i});
    
    % VOCopts.img
    im = imread([imagePath testIms{i}]);
    idx = 1;
    for j=1:length(seg_params.ks)
        k = seg_params.ks(j); % Segmentation threshold k
        minSize = k; % We set minSize = k
        for n = 1:length(seg_params.colorTypes)
            colorType = seg_params.colorTypes{n};
            tic;
            [boxesT{idx} blobIndIm{idx} blobBoxes hierarchy priorityT{idx}]...
                    = Image2HierarchicalGrouping(im, seg_params.sigma, k, minSize, ...
                       colorType, seg_params.simFunctionHandles);
            totalTime = totalTime + toc;
            
            % Get blobs of initial segmentation
            segmentation = SegmentIndices2Blobs(blobIndIm{idx}, blobBoxes);
               
            if length(hierarchy)==1
                tree{idx} = ...
                    RecreateBlobHierarchyLevelsTree(segmentation, ...
                                                    hierarchy{1});
            else
                for h = 1:length(hierarchy)
                    % recreate the tree to be saved
                    tree{h,idx} = ...
                        RecreateBlobHierarchyLevelsTree(segmentation, ...
                                                        hierarchy{h});                
%                 % visualization
%                 subplot(5,5,1)
%                 imagesc(im), axis image
%                 [real_hier, hBlobs] = RecreateBlobHierarchyIndImLevels(blobIndIm, blobBoxes, hierarchy{h});
%                 % visualize if needed
%                 for l = 1:min(length(real_hier),24)
%                     subplot(5,5,l+1), imagesc(real_hier{l}, [0,max(real_hier{end}(:))]),
%                     axis image
%                 end
%                 pause;
%                 clf;
                end
            end
            
            idx = idx + 1;
        end
    end
    
%     % save tree and blobIndIm    
%     for idx = 1:size(tree,2)
%         if idx == 1
%             hdf5write([savePath strtok(testIms{i},'.') '.h5'], [strtok(testIms{i},'.') '/Segm_' num2str(idx)], uint16(blobIndIm{idx}))
%         else
%             hdf5write([savePath strtok(testIms{i},'.') '.h5'], [strtok(testIms{i},'.') '/Segm_' num2str(idx)], uint16(blobIndIm{idx}), 'WriteMode', 'append')
%         end
%         hdf5write([savePath strtok(testIms{i},'.') '.h5'], [strtok(testIms{i},'.') '/leaves_' num2str(idx)], uint16(tree{idx}.leaves), 'WriteMode', 'append');
%         hdf5write([savePath strtok(testIms{i},'.') '.h5'], [strtok(testIms{i},'.') '/nodes_' num2str(idx)], uint16(tree{idx}.nodes), 'WriteMode', 'append');
%     end
    save([savePath strtok(testIms{i},'.') '.mat'],'blobIndIm','tree')
end

% output value
val = 1;

end