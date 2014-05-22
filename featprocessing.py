import numpy as np

class FeatProcessorParams:
    def __init__(self):
        raise NotImplementedError()

class FeatProcessor:
    def __init__(self, params):
        raise NotImplementedError()

    def fit(self, X):
        """ X is [num_examples, num_features] """
        raise NotImplementedError()

    def process(self, X):
        """ Process the examples in X ***in place*** """
        raise NotImplementedError()

    @staticmethod
    def create_feat_processor(params):
        assert isinstance(params, FeatProcessorParams)
        if isinstance(params, FeatProcessorIdentityParams):
            return FeatProcessorIdentity(params)
        elif isinstance(params, FeatProcessorScaleParams):
            return FeatProcessorScale(params)
        else:
            raise ValueError('params instance not recognized')

#=============================================================================

class FeatProcessorIdentityParams(FeatProcessorParams):
    def __init__(self):
        pass

class FeatProcessorIdentity(FeatProcessor):
    def __init__(self, params):
        pass

    def fit(self, X):
        pass

    def process(self, X):
        pass

#=============================================================================

class FeatProcessorScaleParams(FeatProcessorParams):
    def __init__(self, scale=None):
        self.scale = scale

class FeatProcessorScale(FeatProcessor):
    def __init__(self, params):
        assert isinstance(params, FeatProcessorScaleParams)
        self.params = params

    def fit(self, X):
        pass

    def process(self, X):
        assert self.params.scale != None
        X *= self.params.scale
