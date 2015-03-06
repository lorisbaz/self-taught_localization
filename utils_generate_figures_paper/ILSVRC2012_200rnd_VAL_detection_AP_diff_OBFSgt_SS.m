function ILSVRC2012_200rnd_VAL_detection_AP_diff_OBFSgt_SS()

config;

figure('name','ILSVRC2012_200rnd_VAL_detection_AP diff','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Difference in Average Precision');

avg_prec = mean(sort(OBFSgt.average_precision-SS.average_precision))*100;
legend_string = sprintf('%s\n-minus-\n%s\n(AVG=%.4f)', OBFSgt.legend, SS.legend, avg_prec);
this_legend = plot_detection_graph(sort(OBFSgt.average_precision-SS.average_precision), [1:num_classes], 'o', legend_string, OBFSgt.color);
h2=legend('-DynamicLegend'); set(h2,'Interpreter','none');
h2 = legend(this_legend);

legend(h2, 'Location', 'Best');

% draw a vertical line at x=0
line([0, 0], [0, 200]);

end

