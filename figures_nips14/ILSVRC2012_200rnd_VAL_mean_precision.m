function ILSVRC2012_200rnd_VAL_mean_precision()

config;

figure('name','ILSVRC2012_200rnd_VAL_mean_precision','Position', conf.figure_position);
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


% ****** exp14_05stats / mean_precision *** SelectiveSearch, fast   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_precision = [ 0.224900, 0.170650, 0.143067, 0.124075, 0.109400, 0.097567, 0.088486, 0.080687, 0.074711, 0.069870, 0.065482, 0.061892, 0.058608, 0.055779, 0.053140, 0.043740, 0.032560, 0.021602, 0.012270, 0.006828, 0.003041, 0.001622, 0.000942, 0.000811, 0.000788 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp06_21stats_NMS_05 / mean_precision *** exp06_21_NMS_05 (SlidingWindow-heatmap, topC=5)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000 ];
mean_precision = [ 0.178200, 0.152362, 0.135177, 0.122342, 0.110611, 0.101534, 0.094653, 0.088679, 0.083416, 0.078837, 0.075238, 0.072179, 0.069918, 0.067869, 0.066571, 0.062838, 0.061927, 0.061898 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SWheat.legend, 'Color', SWheat.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_06stats_NMS_05 / mean_precision *** exp23_06_NMS_05 (ObfuscationSearch, topC=5)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.478100, 0.295100, 0.218300, 0.174800, 0.146580, 0.126383, 0.111857, 0.100187, 0.091100, 0.083610, 0.077336, 0.071975, 0.067254, 0.063193, 0.059713, 0.046980, 0.033431, 0.021582, 0.011842, 0.006513, 0.004293, 0.004272 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_10stats_NMS_05 / mean_precision *** exp23_10_NMS_05 (ObfuscationSearch, GT)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.462900, 0.292550, 0.218067, 0.174850, 0.146680, 0.126917, 0.112043, 0.100787, 0.091567, 0.084170, 0.077700, 0.072350, 0.067731, 0.063714, 0.060173, 0.047281, 0.033695, 0.021761, 0.011879, 0.006538, 0.004214, 0.004185 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFSgt.legend, 'Color', OBFSgt.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp28_01stats_NMS_05 / mean_precision *** SlidingWindow, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000 ];
mean_precision = [ 0.030600, 0.027450, 0.024333, 0.022125, 0.020260, 0.018683, 0.017286, 0.016337, 0.015578, 0.014710, 0.014191, 0.013692, 0.013154, 0.012771, 0.012333, 0.010484, 0.008190, 0.005881, 0.003565, 0.002097, 0.001008, 0.000645, 0.000625 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SW.legend, 'Color', SW.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

legend(h, 'Location', 'Best');

end
