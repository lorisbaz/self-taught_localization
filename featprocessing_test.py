import numpy as np
import unittest

from featprocessing import *

class FeatProcessorIdentityTest(unittest.TestCase):
    def setUp(self):
        self.params = FeatProcessorIdentityParams()

    def tearDown(self):
        pass

    def test_process(self):
        p = FeatProcessor.create_feat_processor(self.params)
        assert isinstance(p, FeatProcessorIdentity)
        X = np.array([1.0, 2.0, 3.0])
        p.process(X)
        self.assertEqual(X[0], 1.0)
        self.assertEqual(X[1], 2.0)
        self.assertEqual(X[2], 3.0)

#=============================================================================

class FeatProcessorScaleTest(unittest.TestCase):
    def setUp(self):
        self.params = FeatProcessorScaleParams()
        self.params.scale = 2.0

    def tearDown(self):
        pass

    def test_process(self):
        p = FeatProcessor.create_feat_processor(self.params)
        assert isinstance(p, FeatProcessorScale)
        X = np.array([1.0, 2.0, 3.0])
        p.process(X)
        self.assertEqual(X[0], 2.0)
        self.assertEqual(X[1], 4.0)
        self.assertEqual(X[2], 6.0)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
