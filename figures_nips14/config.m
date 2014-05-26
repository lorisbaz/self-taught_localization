MATLAB.Color.yellow = [1 1 0];
MATLAB.Color.magenta = [1 0 1];
MATLAB.Color.orange = [0 1 1];
MATLAB.Color.cyan = [0 1 1]; % cyan
MATLAB.Color.cyanMed = [0 0.6 0.6]; % cyanMed
MATLAB.Color.cyanDark = [0 0.3 0.3]; % cyanDark
MATLAB.Color.red = [1 0 0];
MATLAB.Color.redMed = [0.6 0 0];
MATLAB.Color.redDark = [0.3 0 0];
MATLAB.Color.green = [0 1 0];
MATLAB.Color.blue = [0 0 1];
MATLAB.Color.white = [1 1 1];
MATLAB.Color.black = [0 0 0];
MATLAB.Color.orange = [255 150 0]./255;
MATLAB.Color.orangeLight = [255 170 0]./255;
MATLAB.Color.orangeMed = [255 140 0]./255;
MATLAB.Color.orangeDark = [255 110 0]./255;
MATLAB.Color.greenMed = [45 125 45]./255;
MATLAB.Color.greenDark = [22 62 22]./255; % green medium
MATLAB.Color.greyDark = [90 90 90]./255;
MATLAB.Color.greyMed = [160 160 160]./255;
MATLAB.Color.greyLight = [230 230 230]./255;
MATLAB.Color.orange2 = [225 122 0]./255; % orange
MATLAB.Color.brown = [128 64 0]./255; % brown
MATLAB.Color.brownLight = [255 128 0]./255; % brown light
MATLAB.Color.brownDark = [64 32 0]./255; % brown dark

% size and position of the figures
conf.figure_width = 900;
conf.figure_height = 600;
conf.figure_position = [1 1 conf.figure_width conf.figure_height];

% line width, marker size, font size
conf.lw = 3;
conf.ms = 12;
conf.fs = 18;

% Selective Search
SS.color = MATLAB.Color.greyMed;
SS.legend = 'Selective Search [??]';

% Obfuscation Search (bboxes, TopC=5)
OBFStopC.color = MATLAB.Color.red;
OBFStopC.legend = 'Obfuscation Search bboxes, TopC=5 (our method)';

% Obfuscation Search (bboxes, GT)
OBFSgt.color = MATLAB.Color.redMed;
OBFSgt.legend = 'Obfuscation Search bboxes, GT (our method)';

% SlidingWindow heatmap
SWheat.color = MATLAB.Color.greyMed;
SWheat.legend = 'Sliding Window heatmap';


