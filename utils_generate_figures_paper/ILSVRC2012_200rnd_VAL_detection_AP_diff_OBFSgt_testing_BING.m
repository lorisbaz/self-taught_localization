function ILSVRC2012_200rnd_VAL_detection_AP_diff_OBFSgt_testing_BING()

config;

figure('name','ILSVRC2012_200rnd_VAL_detection_AP diff','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Difference in Average Precision');


idxs = OBFSgt_testing.average_precision~=0;
avg_prec = mean(sort(OBFSgt_testing.average_precision(idxs)-BING.average_precision(idxs)))*100;
% avg_prec = mean(sort(OBFSgt_testing.average_precision-BING.average_precision))*100;
legend_string = sprintf('%s\n-minus-\n%s\n(AVG=%.4f)', OBFSgt_testing.legend, BING.legend, avg_prec);
% this_legend = plot_detection_graph(sort(OBFSgt_testing.average_precision-BING.average_precision), [1:num_classes], 'o', legend_string, OBFSgt_testing.color);
this_legend = plot_detection_graph(sort(OBFSgt_testing.average_precision(idxs)-BING.average_precision(idxs)), [1:sum(idxs)], 'o', legend_string, OBFSgt_testing.color);
h2=legend('-DynamicLegend'); set(h2,'Interpreter','none');
h2 = legend(this_legend);
legend(h2, 'Location', 'Best');

% draw a vertical line at x=0
line([0, 0], [0, 200]);

end

