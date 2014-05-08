%% Create the classid_wnid_words file for ILSVRC2013 detection
clear;

devkit_path = '/home/anthill/vlg/ILSVRC2013_devkit';
dataset_path = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2013/DET';
output_path = './ILSVRC2013_det';
if ~exist(output_path)
    mkdir(output_path)
end

addpath(strcat(devkit_path, '/evaluation'))
class_ids = 1:200;


%% Create classid_wnid_words.txt
ILSVRC2013_det_m = load(strcat(devkit_path, '/data/meta_det.mat'));
fid = fopen(strcat(dataset_path,'/classid_wnid_words.txt'), 'w');
for i = 1:length(ILSVRC2013_det_m.synsets)
    fprintf(fid, '%d\t%s\t%s\n', ILSVRC2013_det_m.synsets(i).ILSVRC2013_DET_ID, ...
                            ILSVRC2013_det_m.synsets(i).WNID, ...
                            ILSVRC2013_det_m.synsets(i).name);
end
fclose(fid);
hash = make_hash(ILSVRC2013_det_m.synsets);


%% Create train*.txt
fprintf('------------TRAINING SET-------------\n')
% Create train_1.txt, train_2.txt, ..., train_200.txt
fid_train = zeros(1, length(class_ids));
for k = class_ids
    fid_train(k) = fopen(strcat(output_path,'/train_',num2str(k),'.txt'), 'w');
end
% Load blacklist
train_blacklist = strcat(devkit_path, '/data/det_lists/train_blacklist.txt');
fid = fopen(train_blacklist);
train_blacklistlist = textscan(fid, '%s', 'Delimiter', ' ');
fclose(fid);
sel_set = 'train/';
for k = class_ids
    % Open training file
    train_file = strcat(devkit_path, '/data/det_lists/train_pos_',num2str(k),'.txt');
    fid = fopen(train_file);
    train_pos_list = textscan(fid, '%s%d', 'Delimiter', ' ');
    fclose(fid);
    train_file = strcat(devkit_path, '/data/det_lists/train_neg_',num2str(k),'.txt');
    fid = fopen(train_file);
    train_neg_list = textscan(fid, '%s%d', 'Delimiter', ' ');
    fclose(fid);
    
    train_all_list{1} = [train_pos_list{1}; train_neg_list{1}];
    train_all_list{2} = [ones(1,length(train_pos_list{2})), -ones(1,length(train_neg_list{2}))];
    t = tic;
    fprintf('Size of the dataset is %d: %d pos and %d neg. \n', ...
                   length(train_all_list{1}), length(train_pos_list{1}), ...
                   length(train_neg_list{1}))
    for i=1:length(train_all_list{1})
        if toc(t) > 10
            fprintf('  Loading on %i of %i\n', i,length(train_all_list{1}));
            t = tic;
        end
        blackimg = strmatch(train_all_list{1}{i}, train_blacklistlist{1}, 'exact'); 
        if ~isempty(blackimg)
            fprintf('Warning image %s is blacklisted.\n', img_path)
            continue;
        end
        folder_name = strtok(train_all_list{1}{i},'_');
        if ~isempty(strfind(folder_name,'ILSVRC2013'))
            folder_name = 'ILSVRC2013_DET_train_extra';
        end
        img_name = [sel_set folder_name '/' train_all_list{1}{i} '.JPEG'];
        img_path = [dataset_path '/' img_name];
        fprintf(fid_train(k), '%s %d \n', img_name, train_all_list{2}(i));
        if ~exist(img_path)
            fprintf('Warning image %s do not exist.\n', img_path)
        end
    end
end
% Close train_1.txt, train_2.txt, ..., train_200.txt
for k = class_ids
    fclose(fid_train(k));
end


%% Create val*.txt
fprintf('-------------VALIDATION SET-------------\n')
% Open validation file
val_file = strcat(devkit_path, '/data/det_lists/val.txt');
fid = fopen(val_file);
validation_list = textscan(fid, '%s%d', 'Delimiter', ' ');
fclose(fid);
% Load blacklist
val_blacklist = strcat(devkit_path, '/data/ILSVRC2013_det_validation_blacklist.txt');
fid = fopen(val_blacklist);
validation_blacklistlist = textscan(fid, '%d%s', 'Delimiter', ' ');
fclose(fid);
img_basenames = validation_list{1};
gt_img_ids = validation_list{2};
gtruth_dir = strcat(dataset_path, '/bbox_val/');
sel_set = 'val/';
t = tic;
% Create val_1.txt, val_2.txt, ..., val_200.txt
fid = zeros(1, length(class_ids));
for k = class_ids
    fid(k) = fopen(strcat(output_path,'/val_',num2str(k),'.txt'), 'w');
end
% Go over the images GT
for i=1:length(img_basenames)
    if toc(t) > 10
        fprintf('  Loading on %i of %i\n', i,length(img_basenames));
        t = tic;
    end
    % Retrieve objects in the blacklist
    blackimg = find(gt_img_ids(i) == validation_blacklistlist{1});
    if isempty(blackimg)
        blackobj = [];
    else
        blackobj = validation_blacklistlist{2}{blackimg};
    end
    % Retrieve groud truth xml 
    img_name = [sel_set img_basenames{i} '.JPEG'];
    img_path = [dataset_path '/' img_name];
    xml_path = [gtruth_dir img_basenames{i} '.xml'];
    if ~exist(img_path)
        fprintf('Warning image %s do not exist.\n', img_path)
    end
    if ~exist(xml_path)
        fprintf('Warning xml %s do not exist.\n', xml_path)
    end
    rec = VOCreadxml(sprintf('%s/%s.xml',gtruth_dir, ...
        img_basenames{i}));
    if ~isfield(rec.annotation,'object')
        continue;
    end
    objects = rec.annotation.object;
    % scan objects
    for j = 1:length(objects)
        if strcmp(objects(j).name, blackobj)
            continue; % just ignore this detection
        end
        id_object = get_class2node(hash, objects(j).name);
        % wrote positive output file
        fprintf(fid(id_object), '%s %d \n', img_name, 1);
        % wrote negative output files
        not_id_object = setdiff(class_ids, id_object);
        for k = not_id_object
            fprintf(fid(k), '%s %d \n', img_name, -1);
        end
    end
       
    if ~isempty(blackobj)
        fprintf('Warning image %s is blacklisted.\n', img_path)
    end
end
% Close val_1.txt, val_2.txt, ..., val_200.txt
for k = class_ids
    fclose(fid(k));
end


%% Create test*.txt
fprintf('-------------TEST SET-------------\n')
test_file = strcat(devkit_path, '/data/det_lists/test.txt');
fid = fopen(test_file);
test_list = textscan(fid, '%s%d', 'Delimiter', ' ');
fclose(fid);
% Create the test file (no Labels)
fid = fopen(strcat(output_path, '/test.txt'), 'w');
sel_set = 'test/';
t = tic;
for i=1:length(test_list{1})
    if toc(t) > 10
        fprintf('  Loading on %i of %i\n', i,length(test_list{1}));
        t = tic;
    end
    img_name = [sel_set test_list{1}{i} '.JPEG'];
    img_path = [dataset_path '/' img_name];
    fprintf(fid, '%s \n', img_name);
    if ~exist(img_path)
        fprintf('Warning image %s do not exist.\n', img_path)
    end
end
fclose(fid);
