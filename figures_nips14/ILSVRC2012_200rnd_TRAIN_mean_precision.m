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
mean_precision = [ 0.520200, 0.319650, 0.234667, 0.185925, 0.154380, 0.132833, 0.116629, 0.104062, 0.094011, 0.085700, 0.078700, 0.072875, 0.067931, 0.063636, 0.059867, 0.046155, 0.031703, 0.019615, 0.010115, 0.005307, 0.003361, 0.003340 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFSgt.legend, 'Color', OBFSgt.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_09stats_NMS_05 / mean_precision *** ObfuscationSearch, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.532500, 0.321750, 0.233033, 0.184225, 0.153120, 0.131433, 0.115200, 0.102987, 0.093078, 0.085020, 0.078100, 0.072325, 0.067385, 0.063086, 0.059380, 0.045740, 0.031460, 0.019501, 0.010043, 0.005285, 0.003413, 0.003398 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp28_02stats_NMS_05 / mean_precision *** Sliding Window, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000 ];
mean_precision = [ 0.046059, 0.035143, 0.029915, 0.025826, 0.022878, 0.020735, 0.018978, 0.017760, 0.016622, 0.015660, 0.014702, 0.013860, 0.013180, 0.012618, 0.012159, 0.010015, 0.007445, 0.005044, 0.002886, 0.001614, 0.000752, 0.000494, 0.000481 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SW.legend, 'Color', SW.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


legend(h, 'Location', 'Best');

end
