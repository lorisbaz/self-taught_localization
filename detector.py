import logging
import numpy as np
import sklearn
import sklearn.cross_validation
from sklearn.cross_validation import KFold
import sklearn.svm
import sklearn.metrics
import sklearn.preprocessing
import sys

from util import *

class DetectorParams:
    def __init__(self):
        raise NotImplementedError()

class Detector:
    """
    This class implement a generic Object Detector.
    """
    def __init__(self):
        self.name = 'Detector' # mandatory field
        raise NotImplementedError()

    def train(self, Xtrain, Ytrain, Xval=[], Yval=[]):
        """
        INPUT:
         Xtrain: ndarray of size [n_samples, n_features]
         Ytrain: ndarray of size [n_samples,], with the class labels {1, -1}
         Xval: this is optional
         Yval: this is optional
        """
        assert isinstance(Xtrain, np.ndarray)
        assert isinstance(Ytrain, np.ndarray)
        assert np.issubdtype(Ytrain.dtype, np.integer)
        assert Xtrain.shape[0] == Ytrain.shape[0]
        self.n_features = Xtrain.shape[1]
        numYtrain = [0, 0]
        for y in Ytrain:
            assert y==1 or y==-1
            numYtrain[(y+1)/2] += 1
        assert numYtrain[0] > 0, 'No negative train labels -1'
        assert numYtrain[1] > 0, 'No positive train labels +1'
        if len(Xval) or len(Yval):
            assert isinstance(Xval, np.ndarray)
            assert isinstance(Yval, np.ndarray)
            assert Xval.shape[0] == Yval.shape[0]
            assert Xtrain.shape[1] == Xval.shape[1]
            numYval = [0, 0]
            for y in Yval:
                assert y==1 or y==-1
                numYval[(y+1)/2] += 1
            assert numYval[0] > 0, 'No negative val labels -1'
            assert numYval[1] > 0, 'No positive val labels +1'

    def predict(self, Xtest):
        """
        INPUT:
         Xtest: ndarray of size [n_samples, n_features] with the test examples
        OUTPUT:
         Spred: ndarray of size [n_samples,] with the predicted scores
        """
        assert isinstance(Xtest, np.ndarray)
        assert self.n_features == Xtest.shape[1]

    @staticmethod
    def create_detector(params):
        assert isinstance(params, DetectorParams)
        if isinstance(params, DetectorFakeParams):
            return DetectorFake(params)
        elif isinstance(params, DetectorLinearSVMParams):
            return DetectorLinearSVM(params)
        else:
            raise ValueError('DetectorParams instance not recognized')

# --------------------------------------------------------------

class DetectorFakeParams(DetectorParams):
    def __init__(self):
        pass

class DetectorFake(Detector):
    def __init__(self, params):
        """ *** PRIVATE CONSTRUCTOR *** """
        pass

    def train(self, Xtrain, Ytrain, Xval=[], Yval=[]):
        # check the input
        Detector.train(self, Xtrain, Ytrain, Xval, Yval)
        pass

    def predict(self, Xtest):
        # check the input
        Detector.predict(self, Xtest)
        # predict
        Spred = np.ones(shape=(Xtest.shape[0],1), dtype=float)
        return Spred

# --------------------------------------------------------------

class DetectorLinearSVMParams(DetectorParams):
    def __init__(self):
        """
        INPUT:
         Call: list of hyperparameters C for the SVM
         numCV: number of CV fold to execute (if at training time the
                validation set is not specifyed)
        """
        self.Call = [10**x for x in range(-4, 3)]
        self.numCV = 5
        self.B = 1.0

class DetectorLinearSVM(Detector):
    """
    Simple implementation of a detector: L2-L1 Batch Linear SVM.
    We use the Average Precision for model selection.
    If the validation set is not specifyed, we do Cross-Validation.
    If the Call list contains a single element, we just train the model
    using that parameter, whether Xval is specified or not.
    """
    def __init__(self, params):
        """ *** PRIVATE CONSTRUCTOR *** """
        self.Call = params.Call
        self.numCV = params.numCV
        self.params = params
        self.svm = None

    def train(self, Xtrain, Ytrain, Xval=[], Yval=[]):
        # check the input
        Detector.train(self, Xtrain, Ytrain, Xval, Yval)
        # for each hyperparameter:
        bestAP = -sys.float_info.max
        bestC = None
        for C in self.Call:
            if len(self.Call) == 1:
                bestC = self.Call[0]
                break
            logging.info('Train C={0}'.format(C))
            # validation mode
            if len(Xval) and len(Yval):
                svm = self.build_svm_(C, self.params.B)
                svm.fit(Xtrain, Ytrain)
                Spred = svm.decision_function(Xval)
                ap = sklearn.metrics.average_precision_score(Yval, Spred)
            # cross-validation mode
            else:
                svm = self.build_svm_(C, self.params.B)
                n_samples = Ytrain.shape[0]
                cv_mode = sklearn.cross_validation.KFold( \
                               n_samples, n_folds=self.numCV, \
                               shuffle=True, random_state=0)
                cv_scores = sklearn.cross_validation.cross_val_score( \
                                svm, Xtrain, Ytrain,
                                scoring='average_precision', cv=cv_mode)
                ap = np.mean(cv_scores)
                logging.info('cv_scores:{0}; ap:{1}'.format(cv_scores, ap))
            # keep the best ap
            if ap > bestAP:
                bestAP = ap
                bestC = C
        # re-train the SVM on the whole dataset using the best C
        self.svm = self.build_svm_(bestC, self.params.B)
        self.svm.fit(Xtrain, Ytrain)
        assert self.svm.classes_[1] == 1

    def predict(self, Xtest):
        # check the input
        Detector.predict(self, Xtest)
        # predict
        Spred = self.svm.decision_function(Xtest)
	return Spred

    @staticmethod
    def build_svm_(C, B):
        svm = sklearn.svm.LinearSVC( \
                penalty='l2', loss='l1', dual=True, tol=0.0001,
                C=C, fit_intercept=True, intercept_scaling=B, \
                class_weight='auto', verbose=0, random_state=0)
        return svm
