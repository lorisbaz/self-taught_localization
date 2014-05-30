function ILSVRC2012_200rnd_VAL_mean_recall()

config;

figure('name','ILSVRC2012_200rnd_VAL_mean_recall','Position', conf.figure_position);
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


% ****** exp14_05stats / mean_recall *** SelectiveSearch, fast   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_recall = [ 0.162616, 0.245926, 0.307500, 0.354514, 0.388487, 0.414413, 0.437003, 0.454217, 0.472229, 0.489498, 0.503264, 0.516739, 0.528609, 0.540384, 0.550614, 0.598311, 0.659556, 0.718882, 0.796641, 0.862426, 0.924993, 0.952894, 0.966196, 0.969432, 0.969673 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp06_21stats_NMS_05 / mean_recall *** exp06_21_NMS_05 (SlidingWindow-heatmap, topC=5)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000 ];
mean_recall = [ 0.125164, 0.211618, 0.281325, 0.338616, 0.381199, 0.418443, 0.452411, 0.480624, 0.503078, 0.520873, 0.537250, 0.550698, 0.563738, 0.572420, 0.582889, 0.605824, 0.611789, 0.611854 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', SWheat.legend, 'Color', SWheat.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_06stats_NMS_05 / mean_recall *** exp23_06_NMS_05 (ObfuscationSearch, topC=5)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_recall = [ 0.351879, 0.432041, 0.474992, 0.503598, 0.524798, 0.540209, 0.554582, 0.565311, 0.576138, 0.585525, 0.593815, 0.600852, 0.606453, 0.612346, 0.618274, 0.641659, 0.673631, 0.709108, 0.756593, 0.794482, 0.840542, 0.841998 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp23_10stats_NMS_05 / mean_recall *** ObfuscationSearch, GT   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_recall = [ 0.339083, 0.425434, 0.471865, 0.501032, 0.522948, 0.540458, 0.554264, 0.567347, 0.577607, 0.587618, 0.595744, 0.602776, 0.609622, 0.615795, 0.621452, 0.643495, 0.676278, 0.713577, 0.759104, 0.801625, 0.865478, 0.869860 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', OBFSgt.legend, 'Color', OBFSgt.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp28_01stats_NMS_05 / mean_recall *** SlidingWindow, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000 ];
mean_recall = [ 0.021431, 0.037567, 0.049603, 0.059719, 0.067987, 0.075290, 0.081113, 0.087289, 0.093312, 0.097808, 0.103665, 0.108682, 0.112520, 0.117061, 0.121215, 0.136355, 0.157485, 0.183314, 0.216552, 0.247016, 0.284081, 0.305698, 0.307066 ];
plot(num_bboxes, mean_recall, '-o', 'DisplayName', SW.legend, 'Color', SW.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

legend(h, 'Location', 'Best');

end
