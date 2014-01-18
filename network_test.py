import os
import network
import unittest


#! /home/anthill/vlg/decaf_131205/bin/python
#/home/data0/vlg/Data/Images/PASCAL_VOC_2007/VOCdevkit/VOC2007/JPEGImages/000001.jpg

def test():
    # load the network model
    print 'Load the network model'
    modeldir = '/home/ironfs/scratch/vlg/Data/Images/'\
        'ILSVRC2012/decaf_model_131205'
    net = DecafNet(modeldir + '/imagenet.decafnet.epoch90', \
		   modeldir + '/imagenet.decafnet.meta')
    # load an example image
    imagefile = '/home/ironfs/scratch/vlg/Data/Images/'\
        'ILSVRC2012/val/ILSVRC2012_val_00000001.JPEG'
    print 'Load the image ' + imagefile
    img = np.asarray(io.imread(imagefile))
    # classification
    scores = net.classify(img)
    print 'Top-5 predicted labels: ' + str(net.top_k_prediction(scores, 5))
    # export the activation values for the fc6 layer (a feature vector)
    scores = net.classify(img, center_only=True)
    feature = net.feature('fc6_cudanet_out')
    print 'The fc6 layer has ' + str(feature.size) + ' neurons'



    imagenet_model_dir = '/home/anthill/aleb/ironfs_vlg/Data/Images'\
        '/ILSVRC2012/decaf_model_131205'
    ilsvrc_image_dir = '/home/anthill/aleb/ironfs_vlg/Data/Images/ILSVRC2012'
    #ilsvrc_image_list = ilsvrc_image_dir + '/val_images_DBG.txt'
    #gt_labels_file = ilsvrc_image_dir + '/val_labels_DBG.txt'
    ilsvrc_image_list = ilsvrc_image_dir + '/val_images.txt'
    gt_labels_file = ilsvrc_image_dir + '/val_labels.txt'
    classid_words_file = ilsvrc_image_dir + '/classid_wnid_words.txt'
    # split the list of images
    fd = open(ilsvrc_image_list, 'r')
    imagefiles = fd.read().split('\n')
    fd.close()
    imagefiles = [x for x in imagefiles if len(x) > 0]




@unittest.skipIf(os.uname()[1] != 'anthill.cs.dartmouth.edu',
                'Skipping TestParFunAnthill because we are not on Anthill')
class NetworkDecaf(unittest.TestCase):
    def setUp(self):
        self.parfun = parfun.ParFunAnthill(sample_function)

    def tearDown(self):
        self.parfun = None

    def test_run1(self):
        self.parfun.add_task(2, 3)
        self.parfun.add_task(2, 2)
        self.parfun.add_task(4, 5)
        out = self.parfun.run()
        self.assertEqual(out[0], 6)
        self.assertEqual(out[1], 4)
        self.assertEqual(out[2], 20)

    def test_run2(self):

#=============================================================================

if __name__ == '__main__':
    unittest.main()
