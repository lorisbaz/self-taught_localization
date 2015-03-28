function ILSVRC2012_200rnd_v_1000_VAL_mean_precision()

config;

figure('name','ILSVRC2012_200rnd_v_1000_VAL_mean_precision','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
xlabel(conf.figure_precision_xlabel);
ylabel(conf.figure_precision_ylabel);
set(gca, 'XScale', 'log');
%title('This is the title');
%set(gca, 'XTick', xvalues);
%axis([1, 50, 0, 1]);

% ****** exp30_08stats_NMS_05 / mean_precision *** ObfuscationSearch similarity + NET features, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.485800, 0.312500, 0.229033, 0.182550, 0.151920, 0.131033, 0.115629, 0.103562, 0.093989, 0.086430, 0.079927, 0.074342, 0.069554, 0.065350, 0.061753, 0.048435, 0.034438, 0.022378, 0.012376, 0.006849, 0.004034, 0.003950 ];
plot(num_bboxes, mean_precision, ['-' OBFStopC_s1.marker], 'DisplayName', OBFStopC_s1.legend, 'Color', OBFStopC_s1.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');

% ****** exp30_10stats_NMS_05 / mean_precision *** ObfuscationSearch similarity + NET features, topC=5   **********
num_bboxes = [ 1.000000, 2.000000, 3.000000, 4.000000, 5.000000, 6.000000, 7.000000, 8.000000, 9.000000, 10.000000, 11.000000, 12.000000, 13.000000, 14.000000, 15.000000, 20.000000, 30.000000, 50.000000, 100.000000, 200.000000, 500.000000, 1000.000000 ];
mean_precision = [ 0.506420, 0.322710, 0.236673, 0.188255, 0.156940, 0.135220, 0.119054, 0.106482, 0.096496, 0.088386, 0.081589, 0.075823, 0.070883, 0.066599, 0.062829, 0.049114, 0.034720, 0.022296, 0.012170, 0.006622, 0.003851, 0.003771 ];
plot(num_bboxes, mean_precision, ['-' OBFStopC_s2.marker], 'DisplayName', OBFStopC_s2.legend, 'Color', OBFStopC_s2.color);
h=legend('-DynamicLegend'); set(h,'Interpreter','none');
set(h,'position',conf.legend_position)

%legend('Location', 'SouthEast');

end
