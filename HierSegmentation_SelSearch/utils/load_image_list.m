% Load image and label lists
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
            if ~exist([savePath labelList_name{i}], 'file')
                mkdir([savePath labelList_name{i}]);
            end
            
        end
        
    case 'test'
        error('Not implemented yet.')
        
    otherwise
        error('Not implemented yet.')
end
