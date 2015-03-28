function ILSVRC2012_200rnd_VAL_detection_AP_diff_OBFSgt_EB()

config;

figure('name','ILSVRC2012_200rnd_VAL_detection_AP diff','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
ylabel('Classes');
xlabel('Difference in Average Precision');

% %%%TMP CODE
% nozero = (EBdet.average_precision~=0);
% EBdet.average_precision = EBdet.average_precision(nozero);
% OBFSgt.average_precision = OBFSgt.average_precision(nozero);
% num_classes = sum(nozero)
% mean(EBdet.average_precision)
% %%%END TMP CODE
avg_prec = mean(sort(OBFSgt.average_precision-EBdet.average_precision))*100;
legend_string = sprintf('%s\n-minus-\n%s\n(AVG=%.4f)', OBFSgt.legend, EBdet.legend, avg_prec);
this_legend = plot_detection_graph(sort(OBFSgt.average_precision-EBdet.average_precision), [1:num_classes], 'o', legend_string, OBFSgt.color);
h2=legend('-DynamicLegend'); set(h2,'Interpreter','none');
h2 = legend(this_legend);

legend(h2, 'Location', 'Best');

% draw a vertical line at x=0
line([0, 0], [0, 200]);

end
