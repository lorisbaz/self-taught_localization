
STL
=====
In this package, we provide the python code for following paper:

[**Self-taught Object Localization using Deep Networks**](http://arxiv.org/abs/1409.3964)
L. Bazzani, A. Bergamo, D. Anguelov, and L. Torresani
*CoRR 2014.*

We included:
* A demo that shows how STL be used to extract the objectness of any image
* The bounding boxes of STL used in our experiments
* The scripts to generate the plots of our paper

Dependencies
------------
* [Python 2.7](https://www.python.org/download/releases/2.7/)
* [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), and [Matplotlib](http://matplotlib.org/) packages.
* [Matlab](http://www.mathworks.com/products/matlab/)
* [Caffe](https://github.com/BVLC/caffe): we used the gist_id [c18d22eb92](https://github.com/BVLC/caffe/tree/c18d22eb92488f02c0256a3fe4ac20a8ad827596) Date: Mon Oct 20 12:57:18 2014 -0700


Getting started
---------------

### Demo

TODO: add documentation

### Bounding Boxes

TODO: add documentation

### Generate Plots Paper

Open Matlab and run the script generate_figures.m.
New curves can be added by modifing the file utils_generate_figures_paper/config.m.

### Additional Info

List of 200 classes randomly selected for the ILSVRC2012-(val,train)-200-RND can be found in the file list_classes_ILSVRC2012-200-RND.txt

L. Bazzani and A. Bergamo contributed equally to the project.
