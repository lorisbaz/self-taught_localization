
STL
=====
In this package, we provide the python code for following paper:

[**Self-taught Object Localization using Deep Networks**](http://arxiv.org/abs/1409.3964).
L. Bazzani, A. Bergamo, D. Anguelov, and L. Torresani
*CoRR 2014.*

We include:
* A demo that shows how STL can be used to extract the objectness bounding boxes of an image
* [TODO] The link of the bounding boxes of STL used in our experiments
* The scripts to generate the plots of our paper

Dependencies
------------
* [Python 2.7](https://www.python.org/download/releases/2.7/),  [NumPy](http://www.numpy.org/), [SciPy](http://www.scipy.org/), [scikit-image](http://scikit-image.org/), and [Matplotlib](http://matplotlib.org/) packages.
* [Matlab](http://www.mathworks.com/products/matlab/)
* [Caffe](https://github.com/BVLC/caffe): installed from the gist_id [c18d22eb92](https://github.com/BVLC/caffe/tree/c18d22eb92488f02c0256a3fe4ac20a8ad827596) Date: Mon Oct 20 12:57:18 2014 -0700
* Segmentation Alg. from the [Selective Search](http://koen.me/research/selectivesearch/) package: install it in the folder img_segmentation/ or use the provided compiled mex

Getting started
---------------

### Demo

* Download the model used in our experiments from [here](https://www.dropbox.com/s/bp24rxbwthonn3g/caffe_model.tar.gz?dl=0)
* Open the file `main_extract_stl.py`
* Change row 19 with the path where you downloaded the model
* Select the option "cpu" or "gpu" at row 21
* Run: `python main_extract_stl.py`

In order to play around with the parameters of STL, open the file `stl_params.py` and look at what you can pass to the the initialization function as argument.

By default the code runs the unsupervised version of STL, but it can be changed to the supervised version by choosing `use_fullimg_GT_label=True`. Note that the label(s) should be provided along with the image at row 24. See the file `prototxt/synset_words.txt` for the list of labels. For the example in the demo, it should be used `gt_labels = ["n01744401"]`.

### Bounding Boxes

TODO: We make available soon the bounding boxes for the dataset we used.

### Generate Plots Paper

Open Matlab and run the script `generate_figures.m`. New curves can be added by modifing the file `utils_generate_figures_paper/config.m`.

### Additional Info

List of 200 classes randomly selected for the ILSVRC2012-(val,train)-200-RND can be found in the file `list_classes_ILSVRC2012-200-RND.txt`

L. Bazzani and A. Bergamo contributed equally to the project.
For the license and usage, have a look at the file LICENSE.