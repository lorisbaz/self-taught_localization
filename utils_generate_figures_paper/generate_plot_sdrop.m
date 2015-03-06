% similarity drop function plot
[X, Y] = meshgrid(0:0.001:1, 0:0.001:1);
s = 1-max(1-X, 1-Y).*abs(X-Y);
figure, mesh(X,Y,s)
figure, imagesc(s), axis equal, axis tight