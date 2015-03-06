function ILSVRC2012_200rnd_VAL_detection_AP_best_worst_categories_top_1()

config;

% ----------- Table top-C, bottom-C classes ----------- %
[~, GTdet.idx_sort] = sort(GTdet.average_precision, 'descend');
[~, OBFSgt.idx_sort] = sort(OBFSgt.average_precision_top_1, 'descend');
[~, SS.idx_sort] = sort(SS.average_precision_top_1, 'descend');

GTdet.average_precision = GTdet.average_precision .* 100;
OBFSgt.average_precision_top_1 = OBFSgt.average_precision_top_1 .* 100;
SS.average_precision_top_1 = SS.average_precision_top_1 .* 100;

topC = 10;
bottomC = 10;
N_words = 1;
factor = 1;
% factor = 100; % if you want AP in percentage, decomment this

fprintf('\\begin{center} \n');
fprintf('\\begin{tabular}{|r|r|r|}\\hline \n');
fprintf(' %s & %s & %s \\\\ \\hline \n', GTdet.legend, OBFSgt.legend, SS.legend);
% top
for k=1:topC
    fprintf('\\best{%s = %1.2f} & \\best{%s = %1.2f} & \\best{%s = %1.2f} \\\\ ', ...
        crop_class_name(classs_description_200{GTdet.idx_sort(k)}, N_words), ...
        GTdet.average_precision(GTdet.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{OBFSgt.idx_sort(k)}, N_words), ...
        OBFSgt.average_precision_top_1(OBFSgt.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{SS.idx_sort(k)}, N_words), ...
        SS.average_precision_top_1(SS.idx_sort(k))*factor);
    fprintf('\n');
end
% separator
%fprintf(' $\\dots$ & $\\dots$ & $\\dots$ \\\\ \n');
% bottom
for k=num_classes-bottomC:num_classes
    fprintf('\\worst{%s = %1.2f} & \\worst{%s = %1.2f} & \\worst{%s = %1.2f} \\\\ ', ...
        crop_class_name(classs_description_200{GTdet.idx_sort(k)}, N_words), ...
        GTdet.average_precision(GTdet.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{OBFSgt.idx_sort(k)}, N_words), ...
        OBFSgt.average_precision_top_1(OBFSgt.idx_sort(k))*factor,...
        crop_class_name(classs_description_200{SS.idx_sort(k)}, N_words), ...
        SS.average_precision_top_1(SS.idx_sort(k))*factor);
    fprintf('\n');
end
fprintf('\\hline \n')
fprintf('mAP = %1.2f & mAP = %1.2f & mAP = %1.2f \\\\ ', ...
        mean(GTdet.average_precision)*factor,...
        mean(OBFSgt.average_precision_top_1)*factor,...
        mean(SS.average_precision_top_1)*factor);
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

