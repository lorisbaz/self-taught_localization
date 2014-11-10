% Speed test for selective search
clear;

dataset_dir = '/home/ironfs/scratch/vlg/Data/Images/PASCAL_VOC_2007/JPEGImages';
file_list = dir([dataset_dir '/*.jpg']);
file_list = file_list(1:100); % select first 100 images

num_files = length(file_list);

time_vec = zeros(1,num_files);
for i = 1:num_files
  file = {[dataset_dir '/' file_list(i).name]};
  fprintf('Image %d/%d: %s', i, num_files, file{1})
  tic;
  selective_search(file, [], 'fast');
  time_vec(i) = toc;
end

final_avg_time = sum(time_vec./num_files);

fprintf('AVG time: %2.4f\n', final_avg_time);
