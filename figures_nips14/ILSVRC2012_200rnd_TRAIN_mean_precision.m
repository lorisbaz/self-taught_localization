function ILSVRC2012_200rnd_TRAIN_mean_precision()

config;

figure('name','ILSVRC2012_200rnd_TRAIN_mean_precision','Position', conf.figure_position);
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


% ****** exp14_07stats / mean_precision *** SelectiveSearch, fast   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_precision = [ 0.245441, 0.187994, 0.155551, 0.132370, 0.115238, 0.101977, 0.091316, 0.083157, 0.076188, 0.070464, 0.065463, 0.061386, 0.057935, 0.054756, 0.051902, 0.041618, 0.029818, 0.019121, 0.010320, 0.005428, 0.002293, 0.001220, 0.000742, 0.000660, 0.000647 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_11stats_NMS_05 / mean_precision *** ObfuscationSearch, GT   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.520500, 0.319650, 0.234633, 0.185975, 0.154460, 0.132883, 0.116671, 0.104125, 0.094044, 0.085740, 0.078727, 0.072892, 0.067946, 0.063650, 0.059880, 0.046160, 0.031710, 0.019617, 0.010116, 0.005308, 0.003362, 0.003340 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFSgt.legend, 'Color', OBFSgt.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% %{'exp23_09stats_NMS_05','ObfuscationSearch, topC=5'}, ...
% TODO

legend(h, 'Location', 'SouthEast');

end
