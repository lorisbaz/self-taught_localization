import skimage
import skimage.transform
import unittest

from configuration import *
from bbox import *
from featextractor import *
from network import *
from annotatedimage import *

class FeatureExtractorNetworkTest(unittest.TestCase):
    def setUp(self):
        # create an AnnotatedImage
        obj1 = AnnotatedObject('label1', None)
        obj1.bboxes.append(BBox(0.11, 0.12, 0.13, 0.14, 0.15))
        obj1.bboxes.append(BBox(0.21, 0.22, 0.23, 0.24, 0.25))
        obj2 = AnnotatedObject('label2', 0.5)
        obj2.bboxes.append(BBox(0.31, 0.32, 0.33, 0.34, None))
        img_anno = AnnotatedImage()
        img_anno.image_name = 'image name'
        img_anno.pred_objects['C1'] = {}
        img_anno.pred_objects['C1']['label1'] = obj1
        img_anno.pred_objects['C1']['label2'] = obj2
        img_anno.pred_objects['C2'] = {}
        img = np.asarray(io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG'))
        width = 100
        height = 200
        img = skimage.transform.resize(img, (height, width))
        img_anno.set_image(img)
        self.img_anno = img_anno
        root = "/home/ironfs/scratch/vlg/Data/Images/ILSVRC2012/caffe_model_141118"
        self.conf = Configuration(root=root)
        # reset the FeatureExtractorNetwork.network_ field
        FeatureExtractorNetwork.network_ = None

    def tearDown(self):
        pass

    def test_init(self):
        # create params
        netparams = NetworkFakeParams()
        params = FeatureExtractorNetworkParams(netparams)
        params.cache_features = True
        # init
        fe = FeatureExtractorNetwork(self.img_anno, params)
        name = 'name:network.NetworkFake-layer:softmax'
        self.assertTrue(name in fe.cache)
        self.assertFalse(fe.cache[name]['featdata'])
        self.assertTrue(fe.cache[name]['featidx'] == {})

    def test_extract(self):
        # create params
        netparams = NetworkFakeParams()
        params = FeatureExtractorNetworkParams(netparams)
        params.cache_features = True
        # init
        fe = FeatureExtractorNetwork(self.img_anno, params)
        name = 'name:network.NetworkFake-layer:softmax'
        # extract
        feats = fe.extract(self.img_anno.pred_objects['C1']['label1'].bboxes)
        self.assertEqual(feats.shape[0], 2)
        self.assertEqual(feats.shape[1], 1000)
        key0 = '11-24-13-28'
        key1 = '21-44-23-48'
        self.assertEqual(len(fe.cache[name]['featidx']), 2)
        self.assertTrue(key0 in fe.cache[name]['featidx'])
        self.assertEqual(fe.cache[name]['featidx'][key0], 0)
        self.assertTrue(key1 in fe.cache[name]['featidx'])
        self.assertEqual(fe.cache[name]['featidx'][key1], 1)
        self.assertIsInstance(fe.cache[name]['featdata'], np.ndarray)
        self.assertEqual(fe.cache[name]['featdata'].shape[0], 2)
        self.assertEqual(fe.cache[name]['featdata'].shape[1], 1000)
        self.assertEqual(fe.cache[name]['featdata'][0,0], 99.0)
        self.assertEqual(fe.cache[name]['featdata'][1,0], 99.0)
        # extract new bboxes
        feats = fe.extract(self.img_anno.pred_objects['C1']['label2'].bboxes)
        self.assertEqual(feats.shape[0], 1)
        self.assertEqual(feats.shape[1], 1000)
        key2 = '31-64-33-68'
        self.assertEqual(len(fe.cache[name]['featidx']), 3)
        self.assertTrue(key2 in fe.cache[name]['featidx'])
        self.assertEqual(fe.cache[name]['featidx'][key2], 2)
        self.assertIsInstance(fe.cache[name]['featdata'], np.ndarray)
        self.assertEqual(fe.cache[name]['featdata'].shape[0], 3)
        self.assertEqual(fe.cache[name]['featdata'].shape[1], 1000)
        self.assertEqual(fe.cache[name]['featdata'][2,0], 99.0)
        # extract the last bbox again
        feats = fe.extract(self.img_anno.pred_objects['C1']['label2'].bboxes)
        self.assertEqual(feats.shape[0], 1)
        self.assertEqual(feats.shape[1], 1000)
        self.assertEqual(len(fe.cache[name]['featidx']), 3)
        self.assertTrue(key2 in fe.cache[name]['featidx'])
        self.assertEqual(fe.cache[name]['featidx'][key2], 2)
        self.assertIsInstance(fe.cache[name]['featdata'], np.ndarray)
        self.assertEqual(fe.cache[name]['featdata'].shape[0], 3)
        self.assertEqual(fe.cache[name]['featdata'].shape[1], 1000)
        self.assertEqual(fe.cache[name]['featdata'][2,0], 99.0)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
