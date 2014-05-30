function ILSVRC2012_1000_VAL_mean_precision()

config;

figure('name','ILSVRC2012_1000_VAL_mean_precision','Position', conf.figure_position);
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


% ****** exp14_06stats / mean_precision *** exp14_06 (SelectiveSearch, fast)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_precision = [ 0.233440, 0.177420, 0.148580, 0.128835, 0.113248, 0.100857, 0.091246, 0.083305, 0.076887, 0.071584, 0.066924, 0.063107, 0.059777, 0.056799, 0.054076, 0.044353, 0.032774, 0.021667, 0.012212, 0.006699, 0.002934, 0.001547, 0.000893, 0.000767, 0.000744 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

% ****** exp23_16stats_NMS_05 / mean_precision *** ObfuscationSearch, TopC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.479860, 0.301970, 0.223987, 0.179745, 0.150680, 0.130363, 0.114997, 0.103105, 0.093551, 0.085702, 0.079127, 0.073543, 0.068728, 0.064527, 0.060836, 0.047556, 0.033528, 0.021413, 0.011537, 0.006277, 0.004028, 0.004001 ];
plot(num_bboxes, mean_precision, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

legend(h, 'Location', 'Best');

end
