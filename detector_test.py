import copy
import math
import numpy as nd
import sklearn
import sklearn.datasets
import unittest

from detector import *

class DetectorLinearSVMTest(unittest.TestCase):
    def setUp(self):
        db = sklearn.datasets.load_iris()
        self.Xtrain = db.data.copy()
        self.Ytrain = db.target.copy()
        for i in range(len(self.Ytrain)):
            if self.Ytrain[i] == 0:
                self.Ytrain[i] = 1
            else:
                self.Ytrain[i] = -1
        self.Xval = self.Xtrain.copy()
        self.Yval = self.Ytrain.copy()

    def tearDown(self):
        pass

    def test_train_val(self):
        params = DetectorLinearSVMParams()
        det = DetectorLinearSVM(params)
        det.train(self.Xtrain, self.Ytrain, self.Xval, self.Yval)
        Spred = det.predict(self.Xval)

    def test_train_cv(self):
        params = DetectorLinearSVMParams()
        det = DetectorLinearSVM(params)
        det.train(self.Xtrain, self.Ytrain)

    def test_train_cv2(self):
        params = DetectorLinearSVMParams()
        params.normalize_features = 'l2'
        for i in range(self.Xtrain.shape[1]):
            self.Xtrain[0,i] = float(i)
        det = DetectorLinearSVM(params)
        det.train(self.Xtrain, self.Ytrain)
        n = math.sqrt(sum( [x*x for x in range(self.Xtrain.shape[1])] ))
        for i in range(self.Xtrain.shape[1]):
            self.assertAlmostEqual(self.Xtrain[0,i], i/float(n))

#=============================================================================

if __name__ == '__main__':
    unittest.main()
