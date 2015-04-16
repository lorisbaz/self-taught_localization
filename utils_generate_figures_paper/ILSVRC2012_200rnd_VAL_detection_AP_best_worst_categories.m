function ILSVRC2012_200rnd_VAL_detection_AP_best_worst_categories()

config;

% ----------- Table top-C, bottom-C classes ----------- %
[~, GTdet.idx_sort] = sort(GTdet.average_precision, 'descend');
[~, OBFSgt.idx_sort] = sort(OBFSgt.average_precision, 'descend');
[~, SS.idx_sort] = sort(SS.average_precision, 'descend');
[~, BING.idx_sort] = sort(BING.average_precision, 'descend');

GTdet.average_precision = GTdet.average_precision .* 100;
OBFSgt.average_precision = OBFSgt.average_precision .* 100;
SS.average_precision = SS.average_precision .* 100;
BING.average_precision = BING.average_precision .* 100;

topC = 10;
bottomC = 5;
N_words = 1;
factor = 1;
% factor = 100; % if you want AP in percentage, decomment this

fprintf('\\begin{center} \n');
fprintf('\\begin{tabular}{|l|p{3.0cm}|p{3.0cm}|p{3.0cm}||p{3.0cm}|}\\hline \n');
fprintf('Positive Boxes + & %s & %s & %s & %s \\\\ \n', BING.legend, SS.legend, OBFSgt.legend, GTdet.legend);
fprintf('Negative/Test Boxes & %s & %s & %s & %s \\\\ \\hline \n',SS.legend,SS.legend,SS.legend,SS.legend)
fprintf('\\hline \n')
fprintf('mAP (all classes) & %1.2f & %1.2f & %1.2f & %1.2f \\\\ ', ...
        mean(BING.average_precision)*factor,...
        mean(SS.average_precision)*factor,...
        mean(OBFSgt.average_precision)*factor,...
        mean(GTdet.average_precision)*factor);
fprintf('\\hline \n')
% top
for k=1:topC
    fprintf('& \\best{%s = %1.2f} & \\best{%s = %1.2f} & \\best{%s = %1.2f} & \\best{%s = %1.2f} \\\\ ', ...
        crop_class_name(classs_description_200{BING.idx_sort(k)}, N_words), ...
        BING.average_precision(BING.idx_sort(k))*factor,... 
        crop_class_name(classs_description_200{SS.idx_sort(k)}, N_words), ...
        SS.average_precision(SS.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{OBFSgt.idx_sort(k)}, N_words), ...
        OBFSgt.average_precision(OBFSgt.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{GTdet.idx_sort(k)}, N_words), ...
        GTdet.average_precision(GTdet.idx_sort(k))*factor);        
    fprintf('\n');
end
% separator
fprintf('\\hline \n')
%fprintf(' $\\dots$ & $\\dots$ & $\\dots$ \\\\ \n');
% bottom
for k=num_classes-bottomC:num_classes
    fprintf('& \\worst{%s = %1.2f} & \\worst{%s = %1.2f} & \\worst{%s = %1.2f} & \\worst{%s = %1.2f} \\\\ ', ...
        crop_class_name(classs_description_200{BING.idx_sort(k)}, N_words), ...
        BING.average_precision(BING.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{SS.idx_sort(k)}, N_words), ...
        SS.average_precision(SS.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{OBFSgt.idx_sort(k)}, N_words), ...
        OBFSgt.average_precision(OBFSgt.idx_sort(k))*factor,...        
        crop_class_name(classs_description_200{GTdet.idx_sort(k)}, N_words), ...
        GTdet.average_precision(GTdet.idx_sort(k))*factor);
    fprintf('\n');
end
fprintf('\\hline \\end{tabular}\n');
fprintf('\\end{center} \n');

% GENERATE A FAKE FIGURE
figure;
text(0.1, 0.1, 'FAKE FIGURE');

end



function class_name_out = crop_class_name(class_name, N)

class_name_out = '';
for i = 1:N
    [this_piece, class_name] = strtok(class_name, ',');
    if ~isempty(this_piece)
        class_name_out = [class_name_out, this_piece];
        if ~isempty(class_name) && (i < N)
            class_name_out = [class_name_out, ', '];
        end
    end
end

end

