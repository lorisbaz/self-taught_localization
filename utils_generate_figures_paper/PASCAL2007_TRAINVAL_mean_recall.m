function PASCAL2007_TRAINVAL_mean_recall()

config;

figure('name','PASCAL2007_TRAINVAL_mean_recall','Position', conf.figure_position);
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


% ****** exp23_08stats_NMS_05 / mean_recall ***  (ObfuscationSearch-boxes, topC=5)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_recall = [ 0.204677, 0.269949, 0.308279, 0.340272, 0.363553, 0.384104, 0.402404, 0.414400, 0.426779, 0.438493, 0.449175, 0.458516, 0.465403, 0.474118, 0.482168, 0.510120, 0.551333, 0.591599, 0.647056, 0.699410, 0.770916, 0.772933 ];
plot(num_bboxes, mean_recall, ['-' OBFStopC.marker], 'DisplayName', OBFStopC.legend, 'Color', OBFStopC.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp14_08stats / mean_recall ***  (SelectiveSearch fast)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000, 3000.000000, 5000.000000 ];
mean_recall = [ 0.095512, 0.151000, 0.195866, 0.229482, 0.254877, 0.277834, 0.294190, 0.315124, 0.332123, 0.346720, 0.361538, 0.377311, 0.391518, 0.406115, 0.418389, 0.467072, 0.539329, 0.608111, 0.715995, 0.810910, 0.905326, 0.946178, 0.964716, 0.967880, 0.968118 ];
plot(num_bboxes, mean_recall, ['-' SS.marker], 'DisplayName', SS.legend, 'Color', SS.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');


% ****** exp29_02stats / mean_recall *** BING (our code)   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000, 2000.000000 ];
mean_recall = [ 0.174096, 0.209681, 0.236915, 0.255477, 0.272941, 0.292083, 0.309075, 0.324505, 0.341502, 0.351664, 0.361089, 0.369596, 0.378222, 0.386282, 0.393932, 0.424864, 0.483356, 0.589425, 0.708100, 0.799721, 0.890425, 0.943949, 0.959432 ];
plot(num_bboxes, mean_recall, ['-' BING.marker], 'DisplayName', BING.legend, 'Color', BING.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
set(h,'position',conf.legend_position)

%legend('Location', 'SouthEast');

end
