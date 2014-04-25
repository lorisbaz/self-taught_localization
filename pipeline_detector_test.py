import unittest

from featextractor import *
from pipeline_detector import *

class PipelineDetectorTest(unittest.TestCase):
    def setUp(self):
        pass
    
    def tearDown(self):
        pass

    def test_init(self):
        category = 'label1'
        params = PipelineDetectorParams()
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.detector_params = DetectorFakeParams()
        params.max_num_neg_bbox_per_image = 5
        pt = PipelineDetector(category, params)        

    def test_train(self):
        category = 'cat'
        params = PipelineDetectorParams()
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.detector_params = DetectorFakeParams()
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pt.train_set = [ \
            PipelineImage('000044', 1, 'test_data/000044.pkl', \
                          FeatureExtractorFakeParams(), 'SELECTIVESEARCH'), \
            PipelineImage('000012', -1, 'test_data/000012.pkl', \
                          FeatureExtractorFakeParams(), 'SELECTIVESEARCH')]
        pt.train()

    def test_evaluate(self):
        # TODO        
        pass
        
    def test_train_elaborate_pos_example(self):
        # PipelineImage and Detector
        category = 'cat'
        params = PipelineDetectorParams()
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.detector_params = DetectorFakeParams()
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 3
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pi = PipelineImage('000044', 1, 'test_data/000044.pkl', \
                           FeatureExtractorFakeParams())
        pi.bboxes = []
        # run - iteration 0
        out = pt.train_elaborate_pos_example_(pi)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].xmin, 0.196)
        self.assertAlmostEqual(out[0].ymin, 0.3003003003)
        self.assertAlmostEqual(out[0].xmax, 0.624)
        self.assertAlmostEqual(out[0].ymax, 0.63963963964)                        
        # run - iteration 1
        pt.iteration = 1
        out = pt.train_elaborate_pos_example_(pi)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].xmin, 0.196)
        self.assertAlmostEqual(out[0].ymin, 0.3003003003)
        self.assertAlmostEqual(out[0].xmax, 0.624)
        self.assertAlmostEqual(out[0].ymax, 0.63963963964)                        
                                
    def test_train_elaborate_neg_example(self):
        pos_bboxes = [BBox(0.10, 0.10, 0.40, 0.60, 0.0), \
                      BBox(0.45, 0.25, 0.80, 0.80, 0.0)]
        bboxes = [[BBox(0.05, 0.05, 0.45, 0.30, 0.1), False], \
                  [BBox(0.50, 0.60, 0.85, 0.85, 0.9), False], \
                  [BBox(0.51, 0.61, 0.86, 0.86, 0.5), False]]
        # PipelineImage and Detector
        category = 'cat'
        params = PipelineDetectorParams()
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.detector_params = DetectorFakeParams()
        params.num_neg_bboxes_to_add_per_image_per_iter = 2
        params.max_num_neg_bbox_per_image = 2
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pi = PipelineImage('000044', 1, 'test_data/000044.pkl', \
                           FeatureExtractorFakeParams())
        pi.bboxes = bboxes
        # run - iteration 0
        pt.train_elaborate_neg_example_(pi)
        self.assertEqual(len([bb for bb in pi.bboxes if bb[1]]), 2)
        # run - iteration 1
        pi.bboxes = [[BBox(0.05, 0.05, 0.45, 0.30, 0.1), False], \
                  [BBox(0.50, 0.60, 0.85, 0.85, 0.9), True], \
                  [BBox(0.51, 0.61, 0.86, 0.86, 0.5), False]]
        pt.iteration = 1
        pt.train_elaborate_neg_example_(pi)
        self.assertEqual(pi.bboxes[0][0].confidence, 0.9)
        self.assertEqual(pi.bboxes[1][0].confidence, 0.5)
        self.assertEqual(pi.bboxes[2][0].confidence, 0.1)                
        self.assertTrue(pi.bboxes[0][1])
        self.assertTrue(pi.bboxes[1][1])
        self.assertFalse(pi.bboxes[2][1])                
                                             
    def test_create_train_buffer(self):
        category = 'label1'
        params = PipelineDetectorParams()
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.detector_params = DetectorFakeParams()
        params.max_num_neg_bbox_per_image = 5
        pt = PipelineDetector(category, params)
        pt.train_set = [ \
            PipelineImage('key1', 1, 'fname1', FeatureExtractorFakeParams()), \
            PipelineImage('key2', 1, 'fname2', FeatureExtractorFakeParams()), \
            PipelineImage('key3', -1,'fname3', FeatureExtractorFakeParams())]
        num_dims = 10
        X, Y = pt.create_train_buffer_(num_dims)
        n = 2*30 + 5*3
        self.assertEqual(X.shape[0], n)
        self.assertEqual(X.shape[1], num_dims)
        self.assertEqual(Y.shape[0], n)
        self.assertEqual(Y.shape[1], 1)
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(Y, np.ndarray))
    
    def test_mark_bboxes_sligtly_overlapping_with_pos_bboxes(self):
        pos_bboxes = [BBox(0.10, 0.10, 0.40, 0.60, 0.0), \
                      BBox(0.45, 0.25, 0.80, 0.80, 0.0)]
        bboxes = [[BBox(0.05, 0.05, 0.45, 0.30, 0), False], \
                  [BBox(0.50, 0.60, 0.85, 0.85, 0), False], \
                  [BBox(0.51, 0.61, 0.86, 0.86, 0), False]]
        max_num_bboxes = 1000
        PipelineDetector.mark_bboxes_sligtly_overlapping_with_pos_bboxes_(\
                   pos_bboxes, bboxes, max_num_bboxes)
        out = bboxes
        self.assertEqual(out[0][0].xmin, 0.05)
        self.assertEqual(out[1][0].xmin, 0.50)
        self.assertEqual(out[2][0].xmin, 0.51)
        self.assertTrue(out[0][1])
        self.assertTrue(out[1][1] or out[2][1])
        self.assertFalse(out[1][1] and out[2][1])

    def test_mark_bboxes_sligtly_overlapping_with_pos_bboxes2(self):
        pos_bboxes = [BBox(0.10, 0.10, 0.40, 0.60, 0.0), \
                      BBox(0.45, 0.25, 0.80, 0.80, 0.0)]
        bboxes = [ [BBox(0.01, 0.01, 0.02, 0.02, 0), False] ]
        max_num_bboxes = 1000
        PipelineDetector.mark_bboxes_sligtly_overlapping_with_pos_bboxes_(\
                   pos_bboxes, bboxes, max_num_bboxes)
        out = bboxes
        self.assertEqual(out[0][0].xmin, 0.01)
        self.assertFalse(out[0][1])
                          
    def test_create_pipeline_images(self):
        # run the code
        key_label_list = [('000012',-1), ('000044',1)]
        params = PipelineDetectorParams()
        params.input_dir = 'test_data'
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.field_name_for_pred_objects_in_AnnotatedImage = 'SELECTIVESEARCH'
        out = PipelineDetector.create_pipeline_images_(key_label_list, params)
        # checks
        self.assertEqual(len(out), 2)
        for o in out:
            self.assertTrue(isinstance(o, PipelineImage))
        self.assertFalse(out[1].bboxes[0][1])
        self.assertEqual(len(out[1].bboxes), 10)
        self.assertAlmostEqual(out[1].bboxes[0][0].xmin, 0.299559471366)
        self.assertAlmostEqual(out[1].bboxes[0][0].ymin, 0.0881057268722)
        self.assertAlmostEqual(out[1].bboxes[0][0].xmax, 0.995594713656)
        self.assertAlmostEqual(out[1].bboxes[0][0].ymax, 0.995594713656)
        self.assertAlmostEqual(out[1].bboxes[0][0].confidence, 0.762684643269)
    
    def test_read_key_label_file(self):
        fname = 'test_data/key_binary_label_file.txt'
        max_pos_examples = 1000
        max_neg_examples = 1000
        out = PipelineDetector.read_key_label_file_( \
                    fname, max_pos_examples, max_neg_examples)
        self.assertEqual(len(out), 20)
        for o in out:
            self.assertEqual(len(o), 2)
        self.assertEqual(out[0][0], '000044')
        self.assertEqual(out[0][1], 1)
        self.assertEqual(out[4][0], '000077')
        self.assertEqual(out[4][1], 1)

    def test_read_key_label_file2(self):
        fname = 'test_data/key_binary_label_file.txt'
        max_pos_examples = 1
        max_neg_examples = 2
        out = PipelineDetector.read_key_label_file_( \
                    fname, max_pos_examples, max_neg_examples)
        self.assertEqual(len(out), 3)
        nposneg = [0, 0]
        for o in out:
            self.assertEqual(len(o), 2)
            nposneg[(o[1]+1)/2] += 1
        self.assertEqual(nposneg[0], 2)
        self.assertEqual(nposneg[1], 1)
                
#=============================================================================

if __name__ == '__main__':
    unittest.main()
