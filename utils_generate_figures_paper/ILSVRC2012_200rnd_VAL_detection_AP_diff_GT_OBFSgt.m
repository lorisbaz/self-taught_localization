function ILSVRC2012_200rnd_VAL_detection_AP_diff_GT_OBFSgt()

config;

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
LEG = cat(1, LEG, [this_legend ' (AVG = ' num2str(mean(sort(GTdet.average_precision-OBFSgt.average_precision))*100) ')']);

h2 = legend(LEG);
legend(h2, 'Location', 'Best');

end

