function this_legend = plot_detection_graph(average_precision, y, marker, legendThis, markerColor)
% *** draw the results

scatter(average_precision, y, 50, marker, 'filled', ...
    'MarkerEdgeColor', markerColor, 'MarkerFaceColor', markerColor)
%legend(legendThis)
this_legend = legendThis;
end