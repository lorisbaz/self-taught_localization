function PASCAL2007_TEST_mean_precision()

config;

figure('name','PASCAL2007_TEST_mean_precision','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
xlabel('Number of bboxes per image');
ylabel('Mean Precision');
set(gca, 'XScale', 'log');
%title('This is the title');
%set(gca, 'XTick', xvalues);
%axis([1, 50, 0, 1]);


% ****** exp14_04stats / precision ***  (SelectiveSearch fast)   **********
% ****** exp14_04stats / mean_precision ***  (SelectiveSearch fast)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_precision = [ 0.138847, 0.110560, 0.093016, 0.082441, 0.073946, 0.067016, 0.061517, 0.057006, 0.053548, 0.050099, 0.047405, 0.044742, 0.042820, 0.041256, 0.039873, 0.033845, 0.026242, 0.018112, 0.010699, 0.006157, 0.002793, 0.001488, 0.000844, 0.000734, 0.000725 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_07stats_NMS_05 / mean_precision *** ObfuscationSearch, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.307076, 0.200268, 0.152813, 0.125232, 0.106329, 0.093561, 0.083424, 0.075337, 0.069076, 0.063941, 0.059710, 0.055937, 0.052487, 0.049705, 0.047075, 0.037712, 0.027287, 0.017949, 0.010031, 0.005518, 0.003532, 0.003526 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


legend(h, 'Location', 'SouthEast');

end
