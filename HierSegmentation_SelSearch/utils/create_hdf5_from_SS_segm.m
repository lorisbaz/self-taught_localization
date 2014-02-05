function success = create_hdf5_from_SS_segm(database_file, imageListClass, savePath)
% Create and save a database HDF5 with an entry for each image in the
% dataset and the respective segmentation results computed using the
% selective search methods, that is based on the Felzenswalb&Huttenlock
% algorithm.
%
% Input:
% - database_file: file name and location of the database
% - imageListClass: list of files in cell array imageListClass{class}{image}
% - savePath: path where the segmentation are saved
%
% Output:
% - file saved on disk at location database_file
% - success: 1 if the computation success, 0 otherwise
%
% Loris Bazzani 02/2014

n_classes = length(imageListClass);

success = 0;

for i = 1:n_classes
    fprintf('-> Class %d / %d \n', i, length(imageListClass));
    for j = 1:size(imageListClass{i})
        fprintf('   Elaborating %d / %d %s\n', j, ...
                 length(imageListClass{i}), imageListClass{i}{j});
        
        % remove file format
        image_name = strtok(imageListClass{i}{j},'.');
        % load the segmentation results
        segm_res = load([savePath image_name '.mat']);
        % go over the different segmentations (for different params)
        for k = 1:length(segm_res.tree)
            % save segmentations (first level)
            if (i==1)&&(j==1)&&(k==1) % first time, create new file
                hdf5write(database_file, ['/' imageListClass{i}{j} ...
                    '/segm_' num2str(k)], uint16(segm_res.blobIndIm{k}));
            else
                hdf5write(database_file, ['/' imageListClass{i}{j} ...
                    '/segm_' num2str(k)], uint16(segm_res.blobIndIm{k}), ...
                    'WriteMode', 'append');
            end
            % ...and trees
            hdf5write(database_file, ['/' imageListClass{i}{j} '/leaves_' ...
                num2str(k)], uint16(segm_res.tree{k}.leaves), ...
                'WriteMode', 'append');
            hdf5write(database_file, ['/' imageListClass{i}{j} ...
                '/nodes_' num2str(k)], uint16(segm_res.tree{k}.nodes), ...
                'WriteMode', 'append');
        end
    end
end

success = 1;
