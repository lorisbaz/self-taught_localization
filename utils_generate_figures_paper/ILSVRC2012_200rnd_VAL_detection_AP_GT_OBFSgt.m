function ILSVRC2012_200rnd_VAL_detection_AP_GT_OBFSgt()

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
set(gca, 'YTick', [0:25:num_classes]);
set(gca, 'XLim', [0 1]);
axis(gca, [0 1 1 num_classes]);
LEG = {};

avg_prec = mean(sort(GTdet.average_precision))*100;
this_legend = plot_detection_graph(GTdet.average_precision(idx), [1:num_classes], 'o', GTdet.legend, GTdet.color);
this_legend = sprintf('%s   mAP = %.2f', this_legend, avg_prec);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

avg_prec = mean(sort(OBFSgt.average_precision))*100;
this_legend = plot_detection_graph(OBFSgt.average_precision(idx), [1:num_classes], 'o', OBFSgt.legend, OBFSgt.color);
this_legend = sprintf('%s   mAP = %.2f', this_legend, avg_prec);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
LEG = cat(1, LEG, this_legend);

h = legend(LEG);
legend(h, 'Location', 'Best');
% set(gca, 'YTickLabel', classs_list_200);
% a = get(gca,'YTickLabel');
% set(gca,'YTickLabel',a,'FontName','Times','fontsize',10)


end
