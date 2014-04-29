import cPickle as pickle
import unittest

from annotatedimage import *
from bbox import *
from heatmap import *
from featextractor import *

class AnnotatedImageTest(unittest.TestCase):
    """
    TODO. Write more tests.
    """
    def setUp(self):
        obj1 = AnnotatedObject('label1', None)
        obj1.bboxes.append(BBox(0.11, 0.12, 0.13, 0.14, 0.15))
        obj1.bboxes.append(BBox(0.21, 0.22, 0.23, 0.24, 0.25))
        obj2 = AnnotatedObject('label2', 0.5)
        obj2.bboxes.append(BBox(0.31, 0.32, 0.33, 0.34, None))
        img_anno = AnnotatedImage()
        img_anno.image_width = 100
        img_anno.image_height = 200
        img_anno.image_name = 'image name'
        img_anno.pred_objects['C1'] = {}
        img_anno.pred_objects['C1']['label1'] = obj1
        img_anno.pred_objects['C1']['label2'] = obj2
        img_anno.pred_objects['C2'] = {}        
        self.img_anno = img_anno

    def tearDown(self):
        self.img_anno = None

    def test_getstate(self):
        # register the extractor first
        params = FeatureExtractorFakeParams()
        self.img_anno.register_feature_extractor(params, True)
        self.assertNotEqual(self.img_anno.feature_extractor_, None)
        # pickle
        s = pickle.dumps(self.img_anno)
        # unpickle
        img_anno = pickle.loads(s)
        # check
        self.assertFalse( hasattr(img_anno, 'feature_extractor_') )
        
    def test_set_image(self):
        img = skimage.io.imread('test_data/ILSVRC2012_val_00000001_n01751748.JPEG')
        skimage.io.imshow(img)
        self.img_anno.set_image(img)
        img2 = self.img_anno.get_image()
        skimage.io.imshow(img2)    
        #skimage.io.show()

    def test_export_pred_bboxes_to_text(self):
        text = self.img_anno.export_pred_bboxes_to_text('C1', 1)
        self.assertEqual(text, \
          'image name\t100\t200\tlabel1\t{0}\t0.21\t0.22\t0.23\t0.24\t0.25\n'\
          'image name\t100\t200\tlabel2\t0.5\t0.31\t0.32\t0.33\t0.34\t{0}\n'\
          .format(-sys.float_info.max, -sys.float_info.max))

    def test_register_feature_extractor(self):
        params = FeatureExtractorFakeParams()
        self.img_anno.register_feature_extractor(params)
        self.assertTrue(hasattr(self.img_anno, 'features'))
        self.assertIsInstance(self.img_anno.feature_extractor_, \
                               FeatureExtractorFake)

    def test_extract_features(self):
        # register the extractor first
        params = FeatureExtractorFakeParams()
        self.img_anno.register_feature_extractor(params, True)
        # extract
        feats = self.img_anno.extract_features( \
                  self.img_anno.pred_objects['C1']['label1'].bboxes)
        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(feats.shape[0], 2)
        self.assertEqual(feats.shape[1], 5)
        self.assertEqual(feats[0,0], 1.0)
        self.assertEqual(self.img_anno.features['FeatureExtractorFake'], 123)
        # extract
        feats = self.img_anno.extract_features( \
                  self.img_anno.pred_objects['C1']['label2'].bboxes)
        self.assertIsInstance(feats, np.ndarray)
        self.assertEqual(feats.shape[0], 1)
        self.assertEqual(feats.shape[1], 5)
        self.assertEqual(feats[0,0], 1.0)
        self.assertEqual(self.img_anno.features['FeatureExtractorFake'], 123)
        
#=============================================================================

if __name__ == '__main__':
    unittest.main()
