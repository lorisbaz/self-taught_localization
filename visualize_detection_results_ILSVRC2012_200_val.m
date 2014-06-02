function visualize_detection_results_ILSVRC2012_200_val()

% hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
% if strcmp(hostname, 'alessandro-Linux')
%   addpath('~/toolbox');
%   load_configuration();
%   addpath([SYSTEM.Code_thirdPart '/VOCdevkit2007/VOCcode']);
% else
%   addpath(['/home/anthill/vlg/VOCdevkit2007/VOCcode']);
% end
%VOCinit;

category_file = 'ILSVRC2013_clsloc/200RND/categories.txt';
fid = fopen(category_file);
classes = textscan(fid, '%s');
fclose(fid);

params = [];
params.quantity = 'average_prec';
params.list_iterations = [2];
params.max_iterations = max(params.list_iterations);
params.classes = classes{1};
params.output_directory = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation';
params.num_classes = length(params.classes);
params.plot_title = 'ILSVR2012 200 validation - exp25_07';
params.prefix_output_files = 'results_detection_ILSVR2012val200';
params.save_output_files = 0;
params.visualize_iter_number = 0;

% *** specify experiments
% params.exps = {...
%     {'exp25_07','GT'}...
%     };
% params.exps = {...
%     {'exp25_09','Obfuscation Search, GT'}...
%     };
params.exps = {...
     {'exp25_10','Selective Search'}...
     };

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
axis(gca, [0 1 1 params.num_classes]);
xlabel(params.quantity ,'Interpreter', 'none');
title([params.plot_title ' - ' params.quantity]   ,'Interpreter', 'none');

% *** draw the results
max_val = zeros(params.num_classes, 1);
last_val = zeros(params.num_classes, 1);
for idxCl = 1:params.num_classes
    for idxIter = params.list_iterations
        if ~isempty(results{idxCl,idxIter})
            eval(['x = results{idxCl,idxIter}.' params.quantity ';']);
            y = idxCl;                        
            if (x > max_val(idxCl))
                max_val(idxCl) = x;
            end
            if idxIter == params.max_iterations
                last_val(idxCl) = x;
            end
            if params.visualize_iter_number
                text(x,y, num2str(idxIter), 'FontSize',14, 'FontWeight','bold');
            else
                scatter(x, y, 50, 'ro', 'filled')
            end
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
text(0.8, 10, sprintf('MAP(max)=%.3f\nMAP=%.3f',mean(max_val),mean(last_val)), 'FontSize',14);    

% print AP
print_matlab_plot_txt(last_val, params.exps{1}, 'average_precision')


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
    for idxIter=opts.list_iterations        
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


function print_matlab_plot_txt(Y, exp, Yquantity)

assert(numel(Y) > 0);
% print header
fprintf('%% ****** %s / %s *** %s   **********\n', exp{1}, Yquantity, exp{2});
% print Y
fprintf('%s = [ ', Yquantity);
for i=1:numel(Y)-1
  fprintf('%f, ' , Y(i));
end
fprintf('%f ];\n' , Y(end));
fprintf('****************************************************\n')

%plot(ntt, acc, '-x', 'DisplayName', classemes.legend, 'Color', classemes.color);
%h=legend('-DynamicLegend'); set(h,'Interpreter','none');

end