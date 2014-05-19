function visualize_detection_results_PASCAL2007_test()

hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
if strcmp(hostname, 'alessandro-Linux')
  addpath('~/toolbox');
  load_configuration();
  addpath([SYSTEM.Code_thirdPart '/VOCdevkit2007/VOCcode']);
else
  addpath(['/home/anthill/vlg/VOCdevkit2007/VOCcode']);
end

VOCinit;

params = [];
params.quantity = 'average_prec';
params.max_iterations = 3;
params.classes = VOCopts.classes;
params.output_directory = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.num_classes = 20;
params.plot_title = 'PASCAL 2007 test set - exp25_05_TEMP10';
params.prefix_output_files = 'results_detection_PASCAL2007test';
params.save_output_files = 0;

% *** specify experiments
params.exps = {{'exp25_05_TEMP10','exp25_05_TEMP10'}};

% *** load results
results = collectResults(params);

% *** figure
h_ap = figure('Position', [1, 1, 800, 400]);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', 1);
set(gcf, 'DefaultLineMarkerSize', 10);
set(gca, 'fontsize', 12);
set(gca, 'YTick', [1:params.num_classes]);
set(gca, 'XLim', [0 1]);
xlabel(params.quantity);
S = get_comparisons();
plot(S.ap_vanDeSande, 1:params.num_classes, '^b');
plot(S.ap_Jasper_RGBSIFT_L3, 1:params.num_classes, '^g');
h = legend('vanDeSande (mAP=0.336)', 'ap_Jasper_RGBSIFT_L3 (mAP=0.2629)');
set(h, 'Interpreter', 'none');
title([params.plot_title ' - ' params.quantity]   ,'Interpreter', 'none');

% *** draw the results
max_val = zeros(params.num_classes, 1);
last_val = zeros(params.num_classes, 1);
for idxCl = 1:params.num_classes
    for idxIter=1:params.max_iterations
        if ~isempty(results{idxCl,idxIter})
            eval(['x = results{idxCl,idxIter}.' params.quantity ';']);
            y = idxCl;                        
            if (x > max_val(idxCl))
                max_val(idxCl) = x;
            end
            if idxIter == params.max_iterations
                last_val(idxCl) = x;
            end
            text(x,y, num2str(idxIter), 'FontSize',14, 'FontWeight','bold');                    
        end
    end
end

% *** ylabel
s = {};
for i=1:params.num_classes
    s{i} = sprintf('(%d) %s (max=%.2f)', i, params.classes{i}, max_val(i));  
end
set(gca, 'YTickLabel', s);

% *** MAP
text(0.6, 1, sprintf('MAP(max)=%.3f\nMAP=%.3f',mean(max_val),mean(last_val)), 'FontSize',14);    

% print last_val
fprintf('last_val = ['); 
for i=1:length(last_val)
   fprintf('%.2f ', last_val(i));
end
fprintf('];\n');

% *** save figures
if params.save_output_files
  saveas(h_ap, [opts.prefix_output_files '_ap.png']);
end


end


function results = collectResults(opts)

assert(numel(opts.exps) == 1); % currently is supported only one experiment at a time
output_directory = [opts.output_directory '/' opts.exps{1}{1}];

finalEval_fileName = [output_directory '/%s/iter_stats%d.mat'];

results = cell(opts.num_classes,  opts.max_iterations);
for idxCl = 1:opts.num_classes
    for idxIter=1:opts.max_iterations        
        fname = sprintf(finalEval_fileName, opts.classes{idxCl}, idxIter-1);
        try        
            S = load(fname);   
            eval(['results{idxCl,idxIter}.' opts.quantity ' = S.' opts.quantity ';']);
            fprintf('[OK] loading %s\n', fname); 
        catch exception
            if strcmp(exception.identifier, 'MATLAB:load:couldNotReadFile')
               fprintf('[ERROR] loading %s\n', fname); 
            else
                %rethrow(exception);
                fprintf('[ERROR] EXCEPTION %s -- %s\n', exception.identifier, exception.message);
            end            
        end
    end
end

end


function S = get_comparisons()

S = [];

S.ap_vanDeSande = [ ...
0.43, ...  % 'aeroplane'
0.46, ...  % 'bicycle'
0.11, ...  % 'bird'
0.11, ...  % 'boat'
0.09, ...  % 'bottle'
0.49, ...  % 'bus'
0.53, ...  % 'car'
0.39, ...  % 'cat'
0.12, ...  % 'chair'
0.36, ...  % 'cow'
0.42, ...  % 'diningtable'
0.26, ...  % 'dog'
0.47, ...  % 'horse'
0.52, ...  % 'motorbike'
0.23, ...  % 'person'
0.11, ...  % 'pottedplant'
0.29, ...  % 'sheep'
0.36, ...  % 'sofa'
0.42, ...  % 'train'
0.48, ...  % 'tvmonitor'
];


% What Jasper obtains with his features, described in the email of Tue, Sep 3, 2013 at 12:08 PM
S.ap_Jasper_RGBSIFT_L3 = [ ...
% aeroplane	
0.314 , ... 
% bicycle	
0.373 , ... 
% bird	
0.059, ... 
% boat	
0.152, ... 
% bottle	
0.045, ... 
% bus	
0.396, ... 
% car	
0.457, ... 
% cat	
0.322, ... 
% chair	
0.046, ... 
% cow	
0.302, ... 
% diningtable	
0.359, ... 
% dog	
0.181, ... 
% horse	
0.392, ... 
% motorbike	
0.387, ... 
% person	
0.123, ... 
% pottedplant	
0.101, ... 
% sheep	
0.269, ... 
% sofa	
0.206, ... 
% train	
0.363, ... 
% tvmonitor	
0.411, ... 
];

end

