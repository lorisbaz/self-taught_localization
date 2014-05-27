function ILSVRC2012_1000_VAL_mean_recall()

config;

figure('name','ILSVRC2012_1000_VAL_mean_recall','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
xlabel('Number of bboxes per image');
ylabel('Mean Recall');
set(gca, 'XScale', 'log');
%title('This is the title');
%set(gca, 'XTick', xvalues);
%axis([1, 50, 0, 1]);


% ****** exp14_06stats / mean_recall *** exp14_06 (SelectiveSearch, fast)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_recall = [ 0.170467, 0.258001, 0.323178, 0.372609, 0.407774, 0.434767, 0.457511, 0.476302, 0.493612, 0.509507, 0.522799, 0.536609, 0.549500, 0.561657, 0.572121, 0.621210, 0.680749, 0.739948, 0.816330, 0.877141, 0.933522, 0.957899, 0.970090, 0.972623, 0.972790 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');



legend(h, 'Location', 'Best');

end
