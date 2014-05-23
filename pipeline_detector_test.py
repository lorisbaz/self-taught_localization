import os
import shutil
import unittest

from featextractor import *
from pipeline_detector import *

class PipelineImageTest(unittest.TestCase):
    def setUp(self):
        neg_bboxes_overlapping_with_pos_params = [0.2, 0.5, 0.5, 0.7]
        self.category = 'cat'
        self.pi_pos = \
            PipelineImage('000044', 1, 'test_data/000044.pkl', \
                FeatureExtractorFakeParams(), 'GT:cat', 'PRED:SELECTIVESEARCH',\
                neg_bboxes_overlapping_with_pos_params)
        self.pi_neg = \
            PipelineImage('000012', -1, 'test_data/000012.pkl', \
                FeatureExtractorFakeParams(), 'GT:cat', 'PRED:SELECTIVESEARCH',\
                neg_bboxes_overlapping_with_pos_params)

    def tearDown(self):
        pass

    def test_train_elaborate_pos_example(self):
        num_neg_bboxes_per_pos_image_during_init = 3
        # run - iteration 0
        iteration = 0
        out = self.pi_pos.train_elaborate_pos_example_( \
                        iteration, num_neg_bboxes_per_pos_image_during_init)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].xmin, 0.196)
        self.assertAlmostEqual(out[0].ymin, 0.3003003003)
        self.assertAlmostEqual(out[0].xmax, 0.624)
        self.assertAlmostEqual(out[0].ymax, 0.63963963964)
        # run - iteration 1
        iteration = 1
        out = self.pi_pos.train_elaborate_pos_example_( \
                        iteration, num_neg_bboxes_per_pos_image_during_init)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out[0].xmin, 0.196)
        self.assertAlmostEqual(out[0].ymin, 0.3003003003)
        self.assertAlmostEqual(out[0].xmax, 0.624)
        self.assertAlmostEqual(out[0].ymax, 0.63963963964)

    def test_train_elaborate_neg_example(self):
        pos_bboxes = [BBox(0.10, 0.10, 0.40, 0.60, 0.0), \
                      BBox(0.45, 0.25, 0.80, 0.80, 0.0)]
        bboxes = [None]*3
        bboxes[0] = BBox(0.05, 0.05, 0.45, 0.30, 0.1)
        bboxes[0].mark = False
        bboxes[1] = BBox(0.50, 0.60, 0.85, 0.85, 0.9)
        bboxes[1].mark = False
        bboxes[2] = BBox(0.51, 0.61, 0.86, 0.86, 0.5)
        bboxes[2].mark = False
        # PipelineImage and Detector
        pi = self.pi_pos
        num_neg_bboxes_to_add_per_image_per_iter = 2
        max_num_neg_bbox_per_image = 2
        pi.get_ai().gt_objects[self.category] = pos_bboxes
        pi.get_ai().pred_objects['SELECTIVESEARCH'] = {}
        anno_obj = AnnotatedObject('cl')
        anno_obj.bboxes = bboxes
        pi.get_ai().pred_objects['SELECTIVESEARCH']['cl'] = anno_obj
        # double-check pi.get_bboxes()
        self.assertEqual(len(pi.get_bboxes()), 3)
        self.assertEqual(pi.get_bboxes()[0].confidence, 0.1)
        self.assertEqual(pi.get_bboxes()[1].confidence, 0.9)
        self.assertEqual(pi.get_bboxes()[2].confidence, 0.5)
        # run - iteration 0
        iteration = 0
        pi.train_elaborate_neg_example_( \
                        iteration, num_neg_bboxes_to_add_per_image_per_iter, \
                        max_num_neg_bbox_per_image)
        self.assertEqual(len([bb for bb in pi.get_bboxes() if bb.mark]), 2)
        # run - iteration 1
        bboxes[0].mark = False
        bboxes[1].mark = True
        bboxes[2].mark = False
        pi.save_marks_and_confidences()
        iteration = 1
        pi.train_elaborate_neg_example_(iteration, \
                    num_neg_bboxes_to_add_per_image_per_iter, \
                    max_num_neg_bbox_per_image)
        self.assertEqual(pi.get_bboxes()[0].confidence, 0.1)
        self.assertEqual(pi.get_bboxes()[1].confidence, 0.9)
        self.assertEqual(pi.get_bboxes()[2].confidence, 0.5)
        self.assertFalse(pi.get_bboxes()[0].mark)
        self.assertTrue(pi.get_bboxes()[1].mark)
        self.assertTrue(pi.get_bboxes()[2].mark)

#==============================================================================

class PipelineDetectorTest(unittest.TestCase):
    def setUp(self):
        # example category
        self.category = 'cat'
        # create some example parameters
        params = PipelineDetectorParams()
        params.input_dir_train = 'test_data'
        params.input_dir_test = 'test_data'
        params.output_dir = 'TEMP_TEST_3475892304'
        if os.path.exists(params.output_dir):
            shutil.rmtree(params.output_dir)
        params.splits_dir = 'test_data'
        params.feature_extractor_params = FeatureExtractorFakeParams()
        params.detector_params = DetectorFakeParams()
        params.field_name_pos_bboxes = 'GT'
        params.field_name_bboxes = 'PRED:SELECTIVESEARCH'
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        self.params = params
        # create some pipeline images
        self.pi_pos = \
            PipelineImage('000044', 1, 'test_data/000044.pkl', \
                FeatureExtractorFakeParams(), 'GT:cat', 'PRED:SELECTIVESEARCH',\
                params.neg_bboxes_overlapping_with_pos_params)
        self.pi_neg = \
            PipelineImage('000012', -1, 'test_data/000012.pkl', \
                FeatureExtractorFakeParams(), 'GT:cat', 'PRED:SELECTIVESEARCH',\
                params.neg_bboxes_overlapping_with_pos_params)
        self.pipeline_images = [self.pi_pos, self.pi_neg]

    def tearDown(self):
        pass

    def test___init(self):
        pt = PipelineDetector(self.category, self.params)

    def test_init_train_evaluate(self):
        category = self.category
        params = self.params
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        params.field_name_for_pred_objects_in_AnnotatedImage = 'SELECTIVESEARCH'
        pt = PipelineDetector(category, params)
        pt.init()
        pt.train_evaluate()

    def test_train(self):
        # single core version
        category = self.category
        params = self.params
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        params.num_cores = 1
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pt.train_set = self.pipeline_images
        pt.train()

    def test_train2(self):
        # multiple cores version
        category = self.category
        params = self.params
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        params.num_cores = 2
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pt.train_set = self.pipeline_images
        pt.train()

    def test_evaluate(self):
        category = self.category
        params = self.params
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pt.test_set = self.pipeline_images
        pt.train_set = pt.test_set
        pt.train()
        # single core
        pt.params.num_cores = 1
        stats = pt.evaluate()

    def test_evaluate2(self):
        category = self.category
        params = self.params
        params.max_num_neg_bbox_per_image = 5
        params.num_neg_bboxes_per_pos_image_during_init = 1
        pt = PipelineDetector(category, params)
        pt.iteration = 0
        pt.test_set = self.pipeline_images
        pt.train_set = pt.test_set
        pt.train()
        # multiple cores
        pt.params.num_cores = 2
        stats = pt.evaluate()

    def test_create_buffer(self):
        num_dims = 10
        n = 22
        X, Y = PipelineDetector.create_buffer_(num_dims, n)
        self.assertEqual(X.shape[0], n)
        self.assertEqual(X.shape[1], num_dims)
        self.assertEqual(Y.size, n)
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(Y, np.ndarray))

    def test_create_train_buffer(self):
        category = 'label1'
        params = self.params
        params.max_num_neg_bbox_per_image = 5
        pt = PipelineDetector(category, params)
        pt.train_set = [self.pi_pos, self.pi_pos, self.pi_neg]
        num_dims = 10
        X, Y = pt.create_train_buffer_(num_dims)
        n = 2*30 + 5*3
        self.assertEqual(X.shape[0], n)
        self.assertEqual(X.shape[1], num_dims)
        self.assertEqual(Y.size, n)
        self.assertTrue(isinstance(X, np.ndarray))
        self.assertTrue(isinstance(Y, np.ndarray))

    def test_mark_bboxes_sligtly_overlapping_with_pos_bboxes(self):
        pos_bboxes = [BBox(0.10, 0.10, 0.40, 0.60, 0.0), \
                      BBox(0.45, 0.25, 0.80, 0.80, 0.0)]
        bboxes = [None]*3
        bboxes[0] = BBox(0.05, 0.05, 0.45, 0.30, 0.0)
        bboxes[0].mark = False
        bboxes[1] = BBox(0.50, 0.60, 0.85, 0.85, 0.0)
        bboxes[1].mark = False
        bboxes[2] = BBox(0.51, 0.61, 0.86, 0.86, 0.0)
        bboxes[2].mark = False
        max_num_bboxes = 1000
        PipelineDetector.mark_bboxes_sligtly_overlapping_with_pos_bboxes_(\
                   pos_bboxes, bboxes, max_num_bboxes)
        out = bboxes
        self.assertEqual(out[0].xmin, 0.05)
        self.assertEqual(out[1].xmin, 0.50)
        self.assertEqual(out[2].xmin, 0.51)
        self.assertTrue(out[0].mark)
        self.assertTrue(out[1].mark or out[2].mark)
        self.assertFalse(out[1].mark and out[2].mark)

    def test_mark_bboxes_sligtly_overlapping_with_pos_bboxes2(self):
        pos_bboxes = [BBox(0.10, 0.10, 0.40, 0.60, 0.0), \
                      BBox(0.45, 0.25, 0.80, 0.80, 0.0)]
        bboxes = [None]
        bboxes[0] = BBox(0.01, 0.01, 0.02, 0.02, 0)
        bboxes[0].mark = False
        max_num_bboxes = 1000
        PipelineDetector.mark_bboxes_sligtly_overlapping_with_pos_bboxes_(\
                   pos_bboxes, bboxes, max_num_bboxes)
        out = bboxes
        self.assertEqual(out[0].xmin, 0.01)
        self.assertFalse(out[0].mark)

    def test_create_pipeline_images(self):
        # run the code
        key_label_list = [('000012',-1), ('000044',1)]
        params = self.params
        out = PipelineDetector.create_pipeline_images_(\
                key_label_list, params, params.input_dir_train, self.category)
        # checks
        self.assertEqual(len(out), 2)
        for o in out:
            self.assertTrue(isinstance(o, PipelineImage))
        self.assertEqual(out[1].key, '000044')
        self.assertEqual(out[1].label, 1)
        self.assertEqual(out[1].fname, 'test_data/000044.pkl')
        self.assertEqual(len(out[1].get_bboxes()), 10)
        self.assertAlmostEqual(out[1].get_bboxes()[0].xmin, 0.299559471366)
        self.assertAlmostEqual(out[1].get_bboxes()[0].ymin, 0.0881057268722)
        self.assertAlmostEqual(out[1].get_bboxes()[0].xmax, 0.995594713656)
        self.assertAlmostEqual(out[1].get_bboxes()[0].ymax, 0.995594713656)
        self.assertAlmostEqual(out[1].get_bboxes()[0].confidence, 0.762684643269)

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
