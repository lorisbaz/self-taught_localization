function ILSVRC2012_200rnd_VAL_detection_AP_OBFSgt_BING_top_1()

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

[average_precision_sort2, idx2] = sort(OBFSgt.average_precision_top_1);
avg_prec = mean(sort(OBFSgt.average_precision_top_1))*100;
this_legend = plot_detection_graph(OBFSgt.average_precision_top_1(idx2), [1:num_classes], 'o', OBFSgt.legend, OBFSgt.color);
this_legend = sprintf('%s   mAP = %.2f', this_legend, avg_prec);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

avg_prec = mean(sort(BING.average_precision_top_1))*100;
this_legend = plot_detection_graph(BING.average_precision_top_1(idx2), [1:num_classes], 'o', BING.legend, BING.color);
this_legend = sprintf('%s   mAP = %.2f', this_legend, avg_prec);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

h = legend(LEG);
legend(h, 'Location', 'Best');

end