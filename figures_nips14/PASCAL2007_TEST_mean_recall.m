function PASCAL2007_TEST_mean_recall()

config;

figure('name','PASCAL2007_TEST_mean_recall','Position', conf.figure_position);
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


% ****** exp14_04stats / mean_recall ***  (SelectiveSearch fast)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_recall = [ 0.100799, 0.160445, 0.201834, 0.238568, 0.265988, 0.288410, 0.308192, 0.324412, 0.342623, 0.355912, 0.369709, 0.380403, 0.394071, 0.407896, 0.421560, 0.473004, 0.546645, 0.621938, 0.722624, 0.817471, 0.908330, 0.948932, 0.965572, 0.968240, 0.968262 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_07stats_NMS_05 / mean_recall *** ObfuscationSearch, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_recall = [ 0.224509, 0.292273, 0.332364, 0.362532, 0.383733, 0.403439, 0.418784, 0.431314, 0.443917, 0.455097, 0.465788, 0.474905, 0.481813, 0.490442, 0.496697, 0.526976, 0.567083, 0.613598, 0.673443, 0.722755, 0.785912, 0.787769 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

legend(h, 'Location', 'SouthEast');

end
