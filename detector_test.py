import copy
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
        self.Xval = db.data.copy()
        self.Yval = db.target.copy()
        for i in range(len(self.Yval)):
            if self.Yval[i] == 0:
                self.Yval[i] = 1
            else:
                self.Yval[i] = -1
        
    def tearDown(self):
        pass

    def test_train(self):
        det = DetectorLinearSVM()
        det.train(self.Xtrain, self.Ytrain, self.Xval, self.Yval)
        det.train(self.Xtrain, self.Ytrain)

#=============================================================================

if __name__ == '__main__':
    unittest.main()
