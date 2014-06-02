function ILSVRC2012_200rnd_VAL_detection_AP()

config;

% -------------- Second mAP graph -------------- %
figure('name','ILSVRC2012_200rnd_VAL_detection_AP','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Average Precision');
set(gca, 'YTick', [0:10:num_classes]);
set(gca, 'XLim', [0 1]);
axis(gca, [0 1 1 num_classes]);
LEG = {};

% ****** exp25_07 / average_precision *** GT   **********
GTdet.average_precision = [ 0.099972, 0.183150, 0.520973, 0.400109, 0.021709, 0.308551, 0.318760, 0.216061, 0.109340, 0.289190, 0.453859, 0.151501, 0.054757, 0.471839, 0.304942, 0.120650, 0.255446, 0.265871, 0.294215, 0.295148, 0.322387, 0.309376, 0.310295, 0.307361, 0.454188, 0.235573, 0.360494, 0.279757, 0.286398, 0.421614, 0.395788, 0.144976, 0.006456, 0.625978, 0.341555, 0.302398, 0.100356, 0.277858, 0.418132, 0.124177, 0.224883, 0.100415, 0.034739, 0.296068, 0.034879, 0.300876, 0.192642, 0.045943, 0.467401, 0.145264, 0.447408, 0.495571, 0.208727, 0.373220, 0.303881, 0.203043, 0.297521, 0.330807, 0.001041, 0.404716, 0.078364, 0.273723, 0.290807, 0.243079, 0.295766, 0.091472, 0.319398, 0.305548, 0.292620, 0.321339, 0.318953, 0.000521, 0.239688, 0.014320, 0.131869, 0.208581, 0.434485, 0.348690, 0.343466, 0.286192, 0.139015, 0.331383, 0.273992, 0.453419, 0.268272, 0.064165, 0.048130, 0.099103, 0.180989, 0.652785, 0.180220, 0.133150, 0.134492, 0.163373, 0.326264, 0.174664, 0.091602, 0.007110, 0.266084, 0.333655, 0.344738, 0.094320, 0.004099, 0.006342, 0.338638, 0.315801, 0.484331, 0.088148, 0.339286, 0.203145, 0.230328, 0.161830, 0.291886, 0.446643, 0.319059, 0.086092, 0.192453, 0.243189, 0.450052, 0.313482, 0.234313, 0.188317, 0.267026, 0.296841, 0.005422, 0.309523, 0.278926, 0.010464, 0.271096, 0.477102, 0.357163, 0.230118, 0.122543, 0.556326, 0.070449, 0.204554, 0.231816, 0.262238, 0.021607, 0.359959, 0.183753, 0.359211, 0.139736, 0.119477, 0.115749, 0.191154, 0.591159, 0.255838, 0.199700, 0.187569, 0.103434, 0.272373, 0.249603, 0.057696, 0.202194, 0.394778, 0.347487, 0.273285, 0.344346, 0.278140, 0.217153, 0.244135, 0.265730, 0.340999, 0.300954, 0.091153, 0.070217, 0.230103, 0.200937, 0.364105, 0.209806, 0.103346, 0.268815, 0.585496, 0.430190, 0.325490, 0.045050, 0.292776, 0.167190, 0.075258, 0.273595, 0.234920, 0.582850, 0.216653, 0.094679, 0.220909, 0.518571, 0.375983, 0.229188, 0.253626, 0.547300, 0.206527, 0.258518, 0.149441, 0.516801, 0.449973, 0.121867, 0.210968, 0.406785, 0.286259 ];
[GT_average_precision_sort, idx] = sort(GTdet.average_precision);
this_legend = plot_detection_graph(GTdet.average_precision(idx), [1:num_classes], 'o', GTdet.legend, GTdet.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

% ****** exp25_09 / average_precision *** Obfuscation Search, GT   **********
OBFSgt.average_precision = [ 0.121032, 0.175403, 0.440779, 0.231068, 0.103935, 0.285048, 0.334567, 0.220359, 0.107820, 0.051836, 0.363916, 0.133506, 0.000670, 0.476139, 0.207333, 0.090344, 0.184634, 0.141136, 0.212381, 0.318171, 0.114040, 0.288300, 0.234048, 0.209240, 0.415584, 0.112427, 0.288177, 0.216799, 0.122647, 0.319999, 0.387109, 0.104506, 0.002695, 0.483932, 0.271117, 0.300004, 0.143756, 0.200228, 0.330163, 0.045817, 0.193767, 0.006924, 0.051777, 0.308918, 0.095436, 0.158987, 0.163432, 0.001542, 0.352497, 0.120226, 0.435556, 0.287688, 0.290662, 0.146292, 0.257152, 0.128022, 0.249278, 0.139374, 0.000297, 0.384756, 0.095049, 0.212501, 0.168924, 0.170344, 0.222287, 0.008939, 0.289272, 0.338219, 0.245917, 0.277998, 0.190328, 0.000190, 0.044813, 0.000962, 0.042726, 0.218777, 0.323521, 0.247562, 0.160168, 0.305688, 0.133007, 0.256742, 0.155437, 0.408654, 0.078304, 0.016815, 0.094199, 0.010904, 0.135203, 0.670697, 0.208324, 0.089818, 0.112302, 0.147461, 0.270920, 0.273329, 0.090920, 0.012688, 0.273284, 0.138860, 0.328683, 0.059260, 0.018189, 0.003697, 0.226838, 0.278338, 0.429078, 0.071896, 0.217251, 0.135474, 0.198704, 0.156453, 0.268656, 0.499858, 0.237040, 0.074013, 0.140484, 0.096262, 0.305598, 0.236787, 0.190876, 0.292157, 0.159197, 0.121515, 0.000337, 0.197770, 0.211130, 0.022920, 0.334222, 0.305723, 0.389506, 0.223226, 0.144176, 0.542979, 0.021286, 0.061349, 0.156204, 0.145901, 0.016255, 0.326140, 0.129327, 0.388203, 0.138551, 0.059881, 0.106888, 0.076964, 0.535866, 0.188197, 0.057388, 0.181940, 0.107703, 0.278589, 0.199928, 0.014965, 0.163177, 0.108079, 0.313699, 0.295742, 0.304459, 0.174602, 0.177189, 0.123076, 0.199412, 0.222553, 0.279584, 0.090995, 0.012672, 0.151582, 0.167787, 0.317868, 0.130643, 0.041860, 0.208799, 0.500659, 0.314433, 0.256988, 0.048159, 0.316044, 0.153760, 0.134648, 0.249892, 0.179291, 0.519688, 0.202073, 0.046769, 0.130718, 0.420011, 0.292591, 0.311753, 0.084282, 0.278543, 0.210247, 0.059291, 0.154310, 0.441775, 0.343645, 0.093420, 0.243542, 0.379366, 0.253393 ];
this_legend = plot_detection_graph(OBFSgt.average_precision(idx), [1:num_classes], 'o', OBFSgt.legend, OBFSgt.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

h = legend(LEG);
legend(h, 'Location', 'Best');
% set(gca, 'YTickLabel', classs_list_200);
% a = get(gca,'YTickLabel');
% set(gca,'YTickLabel',a,'FontName','Times','fontsize',10)


% -------------- Second mAP graph -------------- %
figure('name','ILSVRC2012_200rnd_VAL_detection_AP','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Average Precision');
set(gca, 'YTick', [0:10:num_classes]);
set(gca, 'XLim', [0 1]);
axis(gca, [0 1 1 num_classes]);
LEG = {};

% ****** exp25_09 / average_precision *** Obfuscation Search, GT   **********
[average_precision_sort2, idx2] = sort(OBFSgt.average_precision);
this_legend = plot_detection_graph(OBFSgt.average_precision(idx2), [1:num_classes], 'o', OBFSgt.legend, OBFSgt.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

% -----> NOTE: THOSE ARE PARTIAL RESULTS: USE THE FINAL ONES!!!
% ****** exp25_10 / average_precision *** Selective Search   **********
SS.average_precision = [ 0.160044, 0.000000, 0.353962, 0.000000, 0.000000, 0.000000, 0.301131, 0.000000, 0.103066, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.204948, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.189406, 0.223920, 0.382654, 0.000000, 0.298591, 0.249072, 0.103115, 0.000000, 0.337685, 0.000000, 0.002226, 0.369521, 0.319336, 0.000000, 0.140378, 0.000000, 0.000000, 0.000000, 0.185807, 0.000000, 0.000000, 0.000000, 0.106015, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.343587, 0.000000, 0.245163, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000080, 0.000000, 0.105125, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.042597, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.183991, 0.000000, 0.000000, 0.000000, 0.094643, 0.000000, 0.094056, 0.000000, 0.000000, 0.000000, 0.113366, 0.000000, 0.297778, 0.000000, 0.090944, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.006454, 0.000000, 0.201835, 0.323074, 0.412085, 0.000000, 0.251887, 0.000000, 0.152209, 0.145416, 0.274267, 0.000000, 0.184216, 0.100077, 0.199999, 0.000000, 0.219124, 0.000000, 0.151780, 0.000000, 0.169095, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.279288, 0.000000, 0.314222, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.120099, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.128264, 0.000000, 0.285906, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.203607, 0.000000, 0.140248, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.018421, 0.000000, 0.155222, 0.000000, 0.000000, 0.000000, 0.508628, 0.000000, 0.000000, 0.000000, 0.259992, 0.000000, 0.000000, 0.000000, 0.323770, 0.000000, 0.000000, 0.000000, 0.441907, 0.000000, 0.000000, 0.000000, 0.308388, 0.000000 ];
this_legend = plot_detection_graph(SS.average_precision(idx2), [1:num_classes], 'o', SS.legend, SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

h = legend(LEG);
legend(h, 'Location', 'Best');



% -------------- Difference AP graphs --------------- %
figure('name','ILSVRC2012_200rnd_VAL_detection_AP diff','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Difference in Average Precision');
LEG = {};

this_legend = plot_detection_graph(sort(GTdet.average_precision-OBFSgt.average_precision), [1:sum(OBFSgt.average_precision>0)], 'o', [GTdet.legend ' - ' OBFSgt.legend], OBFSgt.color);
h2=legend('-DynamicLegend'); set(h2,'Interpreter','none');
LEG = cat(1, LEG, [this_legend ' (AVG = ' num2str(mean(sort(GTdet.average_precision-OBFSgt.average_precision))) ')']);

h2 = legend(LEG);
legend(h2, 'Location', 'Best');

figure('name','ILSVRC2012_200rnd_VAL_detection_AP diff','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Difference in Average Precision');
LEG = {};

this_legend = plot_detection_graph(sort(SS.average_precision(SS.average_precision>0)-OBFSgt.average_precision(SS.average_precision>0)), [1:sum(SS.average_precision>0)], 'o', [SS.legend ' - ' OBFSgt.legend], OBFSgt.color);
h2=legend('-DynamicLegend'); set(h2,'Interpreter','none');
LEG = cat(1, LEG, [this_legend ' (AVG = ' num2str(mean(sort(SS.average_precision(SS.average_precision>0)-OBFSgt.average_precision(SS.average_precision>0)))) ')']);

h2 = legend(LEG);
legend(h2, 'Location', 'Best');

% ----------- Table top-C, bottom-C classes ----------- %
[~, GTdet.idx_sort] = sort(GTdet.average_precision, 'descend');
[~, OBFSgt.idx_sort] = sort(OBFSgt.average_precision, 'descend');
[~, SS.idx_sort] = sort(SS.average_precision, 'descend');

topC = 10;
bottomC = 10;
N_words = 1;

fprintf('\\begin{center} \\footnotesize \n');
fprintf('\\begin{tabular}{|r|r|r|}\\hline \n');
fprintf(' %s & %s & %s \\\\ \\hline \n', GTdet.legend, OBFSgt.legend, SS.legend);
% top
for k=1:topC
    fprintf('%s = %1.4f & %s = %1.4f & %s = %1.4f \\\\ ', ...
        crop_class_name(classs_description_200{GTdet.idx_sort(k)}, N_words), ...
        GTdet.average_precision(GTdet.idx_sort(k)),...
        crop_class_name(classs_description_200{OBFSgt.idx_sort(k)}, N_words), ...
        OBFSgt.average_precision(OBFSgt.idx_sort(k)),...
        crop_class_name(classs_description_200{SS.idx_sort(k)}, N_words), ...
        SS.average_precision(SS.idx_sort(k)));
    fprintf('\n');
end
% bottom
fprintf(' $\\dots$ & $\\dots$ & $\\dots$ \\\\ \n');
for k=num_classes-bottomC:num_classes
    fprintf('%s = %1.4f & %s = %1.4f & %s = %1.4f \\\\ ', ...
        crop_class_name(classs_description_200{GTdet.idx_sort(k)}, N_words), ...
        GTdet.average_precision(GTdet.idx_sort(k)),...
        crop_class_name(classs_description_200{OBFSgt.idx_sort(k)}, N_words), ...
        OBFSgt.average_precision(OBFSgt.idx_sort(k)),...
        crop_class_name(classs_description_200{SS.idx_sort(k)}, N_words), ...
        SS.average_precision(SS.idx_sort(k)));
    fprintf('\n');
end
fprintf('\\hline \n')
fprintf('AVG = %1.4f & AVG = %1.4f & AVG = %1.4f \\\\ ', ...
        mean(GTdet.average_precision),...
        mean(OBFSgt.average_precision),...
        mean(SS.average_precision));
fprintf('\\hline \\end{tabular}\n');
fprintf('\\end{center} \n');

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

function this_legend = plot_detection_graph(average_precision, y, marker, legendThis, markerColor)
% *** draw the results

scatter(average_precision, y, 50, marker, 'filled', ...
    'MarkerEdgeColor', markerColor, 'MarkerFaceColor', markerColor)
%legend(legendThis)
this_legend = legendThis;
end