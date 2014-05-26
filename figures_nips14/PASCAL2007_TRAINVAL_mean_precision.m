function PASCAL2007_TRAINVAL_mean_precision()

config;

figure('name','PASCAL2007_TRAINVAL_mean_precision','Position', conf.figure_position);
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


% ****** exp14_08stats / mean_precision ***  (SelectiveSearch fast)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_precision = [ 0.131104, 0.103970, 0.090576, 0.079862, 0.071128, 0.064918, 0.059000, 0.055330, 0.051933, 0.048843, 0.046479, 0.044444, 0.042603, 0.041042, 0.039579, 0.033321, 0.025784, 0.017597, 0.010525, 0.006054, 0.002761, 0.001471, 0.000836, 0.000725, 0.000716 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_08stats_NMS_05 / mean_precision ***  (ObfuscationSearch-boxes, topC=5)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.280615, 0.185418, 0.141736, 0.117292, 0.100262, 0.088371, 0.079654, 0.072008, 0.066016, 0.061219, 0.057186, 0.053660, 0.050320, 0.047619, 0.045303, 0.036185, 0.026315, 0.017111, 0.009506, 0.005301, 0.003459, 0.003457 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

legend(h, 'Location', 'SouthEast');

end
