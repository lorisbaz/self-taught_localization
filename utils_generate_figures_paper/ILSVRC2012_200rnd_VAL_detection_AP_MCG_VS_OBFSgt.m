function ILSVRC2012_200rnd_VAL_detection_AP_MCG_VS_OBFSgt()

config;

% -------------- Difference AP graphs --------------- %
figure('name','ILSVRC2012_200rnd_VAL_detection_AP diff','Position', conf.figure_position);
hold on;
grid on;
set(gcf, 'DefaultLineLineWidth', conf.lw);
set(gcf, 'DefaultLineMarkerSize', conf.ms);
set(gca, 'fontsize', conf.fs);
xlabel(['AP - ' MCGdet.legend '+SelSearch']);
ylabel(['AP - STL-CL+SelSearch']);
%ylabel(['AP - STL$_{CL}$+SelSearch'],'interpreter','latex');
LEG = {};
plot([0,0.7],[0,0.7])
%%%TMP CODE
nozero = (MCGdet.average_precision~=0);
MCGdet.average_precision = MCGdet.average_precision(nozero);
OBFSgt.average_precision = OBFSgt.average_precision(nozero);
%%%END TMP CODE
idx_gt = (MCGdet.average_precision-OBFSgt.average_precision)>0;
scatter(MCGdet.average_precision(idx_gt),  OBFSgt.average_precision(idx_gt), 65, 'o', 'filled', ...
    'MarkerEdgeColor', OBFSgt.color, 'MarkerFaceColor', OBFSgt.color)
idx_obf = ~(idx_gt); sum(idx_obf)
scatter(MCGdet.average_precision(idx_obf), OBFSgt.average_precision(idx_obf), 65, 'o', 'filled', ...
    'MarkerEdgeColor', OBFSgt.color, 'MarkerFaceColor', OBFSgt.color)
% h2=legend('-DynamicLegend'); set(h2,'Interpreter','none');
% LEG = cat(1, LEG, [GTdet.legend ' is better']);
% LEG = cat(1, LEG, [OBFSgt.legend ' is better']);

% h2 = legend(LEG);
% legend(h2, 'Location', 'Best');
axis equal;
axis([0, 0.7, 0, 0.7]);
end