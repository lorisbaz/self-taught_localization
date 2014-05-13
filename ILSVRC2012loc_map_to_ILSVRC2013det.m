%% Compute stats and mapping ILSVRC2013_det->ILSVRC2012_loc
clear;
devkit_path = '/home/anthill/vlg/ILSVRC2013_devkit';
dataset12_path = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012';
dataset13_path = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2013/DET';
output_path = dataset13_path;

% load synsets
ILSVRC2013_det_m = load(strcat(devkit_path, '/data/meta_det.mat'));

% Open the ILSVRC2012 class list file
fid = fopen(strcat(dataset12_path, '/classid_wnid_words.txt'));
ILSVRC2012_loc = textscan(fid, '%d%s%s', 'Delimiter', '\t');
fclose(fid);

% Open the ILSVRC2013 class list file (created above)
fid = fopen(strcat(dataset13_path, '/classid_wnid_words.txt'));
ILSVRC2013_det = textscan(fid, '%d%s%s', 'Delimiter', '\t');
fclose(fid);


% Check ILSVRC2013_det overlap with ILSVRC2012_loc (direct/children/anchestor)
class_annotated = 1:200;
% check overlapping
ILSVRC2013_det_200 = ILSVRC2013_det_m.synsets(class_annotated);
overlap_200 = zeros(1,length(class_annotated));
mapping_overlap_200(length(class_annotated))=struct('type',[],'ILSVRC2012_loc',[],'ILSVRC2013_det',[]);
for i = 1:length(ILSVRC2013_det_200)
    for j = 1:length(ILSVRC2012_loc{1})
        if strcmp(ILSVRC2013_det_200(i).WNID,ILSVRC2012_loc{2}{j})
            overlap_200(i) = 1;
            mapping_overlap_200(i).type = 'direct';
            mapping_overlap_200(i).ILSVRC2012_loc{1} = ILSVRC2012_loc{2}{j}; 
            mapping_overlap_200(i).ILSVRC2013_det{1} = ILSVRC2013_det_200(i).WNID;
            break;
        end
    end
end
fprintf('Overlapping first 200 classes: %1.2f \n', sum(overlap_200)/length(overlap_200))

% check if the children are covered in ILSVRC2012_loc
for i = 1:length(overlap_200)
    if overlap_200(i)==0
        jj = 1;
        for j = ILSVRC2013_det_200(i).children
            for k = 1:length(ILSVRC2012_loc{1})
                if strcmp(ILSVRC2013_det_m.synsets(j).WNID,ILSVRC2012_loc{2}{k})
                    overlap_200(i) = 1;
                    mapping_overlap_200(i).type = 'children';
                    mapping_overlap_200(i).ILSVRC2012_loc{jj} = ILSVRC2012_loc{2}{k}; 
                    mapping_overlap_200(i).ILSVRC2013_det{1} = ILSVRC2013_det_200(i).WNID;
                    jj = jj + 1;
                    %break;
                end
            end
        end
    end
end
fprintf('Overlapping between the remaining classes and children: %1.2f \n', ...
                    sum(overlap_200)/length(overlap_200))

                
% check if the fathers are covered in ILSVRC2012_loc
for i = 1:length(overlap_200)
    if overlap_200(i)==0
        jj = 1;
        for synsets_m = ILSVRC2013_det_m.synsets
            for j = synsets_m.children
                if strcmp(ILSVRC2013_det_m.synsets(j).WNID,ILSVRC2013_det_200(i).WNID) && ...
                                j~=i % father found
                    for k = 1:length(ILSVRC2012_loc{1})
                        if strcmp(ILSVRC2013_det_m.synsets(j).WNID,ILSVRC2012_loc{2}{k})
                            overlap_200(i) = 1;
                            mapping_overlap_200(i).type = 'father';
                            mapping_overlap_200(i).ILSVRC2012_loc{jj} = ILSVRC2012_loc{2}{k};
                            mapping_overlap_200(i).ILSVRC2013_det{1} = ILSVRC2013_det_200(i).WNID;
                            jj = jj + 1;
                            %break;
                        end
                    end
                end
            end
        end
    end
end
fprintf('Overlapping between the remaining classes and anchestors: %1.2f \n', ...
                    sum(overlap_200)/length(overlap_200))                

fprintf('Not covered classes: \n');
j = find(overlap_200==0);
{ILSVRC2013_det_m.synsets(j).name}


% Generate the subset that exactly overlaps with ILSVRC2012 ('direct')
fid = fopen(strcat(output_path, '/label_subset_overlap_with_ILSVRC2012.txt'), 'w');
for i = 1:length(mapping_overlap_200)
    if strcmp(mapping_overlap_200(i).type, 'direct')
       fprintf(fid, '%d\t%s\t%s\n', i, mapping_overlap_200(i).ILSVRC2013_det{1}, ILSVRC2013_det_m.synsets(i).name); 
    end
end
fclose(fid);