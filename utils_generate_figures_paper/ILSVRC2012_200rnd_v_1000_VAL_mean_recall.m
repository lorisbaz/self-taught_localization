function ILSVRC2012_200rnd_v_1000_VAL_mean_recall()

config;

figure('name','ILSVRC2012_200rnd_v_1000_VAL_mean_recall','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
xlabel(conf.figure_recall_xlabel);
ylabel(conf.figure_recall_ylabel);
set(gca, 'XScale', 'log');
%title('This is the title');
%set(gca, 'XTick', xvalues);
%axis([1, 50, 0, 1]);


% ****** exp30_08stats_NMS_05 / mean_recall *** ObfuscationSearch similarity + NET features, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_recall = [ 0.356513, 0.455985, 0.497552, 0.525479, 0.543791, 0.559967, 0.574301, 0.585245, 0.595455, 0.606247, 0.615212, 0.622692, 0.629747, 0.635345, 0.641567, 0.664812, 0.697646, 0.740067, 0.794518, 0.844824, 0.899149, 0.904258 ];
plot(num_bboxes, mean_recall, ['-' OBFStopC_s1.marker], 'DisplayName', OBFStopC_s1.legend, 'Color', OBFStopC_s1.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp30_10stats_NMS_05 / mean_recall *** ObfuscationSearch similarity + NET features, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_recall = [ 0.377169, 0.477222, 0.521555, 0.549911, 0.570349, 0.587273, 0.601085, 0.612298, 0.622261, 0.631556, 0.639678, 0.646979, 0.653938, 0.660155, 0.665991, 0.688544, 0.720843, 0.759103, 0.809899, 0.853417, 0.902924, 0.907974 ];
plot(num_bboxes, mean_recall, ['-' OBFStopC_s2.marker], 'DisplayName', OBFStopC_s2.legend, 'Color', OBFStopC_s2.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
set(h,'position',conf.legend_position)

%legend('Location', 'SouthEast');

end
