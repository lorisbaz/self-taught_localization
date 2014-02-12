% Configurations
fprintf('---Configure computation---\n')
rootPath = pwd;
toolboxPath = '/home/anthill/aleb/clients/aleb/aleb/toolbox/';
hostname = char( getHostName( java.net.InetAddress.getLocalHost ) );
switch (hostname)
    case 'anthill.cs.dartmouth.edu'
        imagePath = '/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012/';
        trainPath = [imagePath 'train/'];
        valPath = [imagePath 'val/'];
        testPath = [imagePath 'test/'];
        savePath  = '/home/ironfs/scratch/vlg/Data_projects/grayobfuscation/segment_ILSVRC2012/';
        selectivePath = '/home/anthill/vlg/SelectiveSearchCodeIJCV/';        
        addpath(toolboxPath)
        
        run_on_anthill = 1;
        
    case 'alessandro-Linux'
        imagePath = '/home/alessandro/Data/ILSVRC2012/';
        trainPath = [imagePath 'train/'];
        valPath = [imagePath 'val/'];
        testPath = [imagePath 'test/'];
        savePath  = 'TODO';
        selectivePath = 'TODO';
        
        run_on_anthill = 0;
        
    case 'lbazzani-desk'
        imagePath = '/home/lbazzani/DATASETS/ILSVRC2012/';
        trainPath = [imagePath 'train/'];
        valPath = [imagePath 'val/'];
        testPath = [imagePath 'test/'];
        savePath  = '/home/lbazzani/CODE/DATA/ILSVRC2012/segmentation/';
        selectivePath = '/home/lbazzani/CODE/3rd_part_libs/SelectiveSearchCodeIJCV/';

        run_on_anthill = 0;

end
% add required libraires
addpath([rootPath '/utils']);
addpath(genpath(selectivePath));