from annotatedimage import *
from bbox import *
from network import *

class FeatureExtractorParams():
    """
    Base class for the parameters.
    The subclass must have the field 'name'.
    """
    def __init__(self):
        raise NotImplementedError()
        
class FeatureExtractor():
    """
    Extractor features from an AnnotatedImage.
    You must construct the objects through the method 'create_instance',
    and extract the features using the method 'extract'
    """

    def __init__(self):
        """
        The constructors for all the FeatureExtractors is private.
        """
        self.name = 'FeatureExtractor' # mandatory field
        raise NotImplementedError()

    def extract(self, bboxes):
        """
        The base method to extract the features.
        Normally, it should return a ndarray of size [len(bboxes), num_features]
        """
        raise NotImplementedError()

    def get_cache(self):
        """
        Returns a Python object containing feature-dependent data, useful
        to speed-up future feature extraction calls.
        """
        raise NotImplementedError()

    @staticmethod
    def create_feature_extractor(anno_image, params):
        """
        Factory for the FeatureExtractors, taking an AnnotatedImage and a
        FeatureExtractorParams instance as input
        """
        assert isinstance(params, FeatureExtractorParams)
        if isinstance(params, FeatureExtractorNetworkParams):
            return FeatureExtractorNetwork(anno_image, params)
        elif isinstance(params, FeatureExtractorFakeParams):
            return FeatureExtractorFake()
        else:
            raise ValueError('FeatureExtractorParams instance not recognized')
    
#=============================================================================

class FeatureExtractorFakeParams(FeatureExtractorParams):
    def __init__(self):
        pass
        
class FeatureExtractorFake(FeatureExtractor):
    """
    Fake feature extractor module, for debugging purposes.
    """

    def __init__(self):
        """  *** PRIVATE CONSTRUCTOR *** """
        self.name = 'FeatureExtractorFake'
        self.num_feats = 5
    
    def extract(self, bboxes):
        """
        Return a matrix of ones.
        """
        return np.ones(shape=(len(bboxes), self.num_feats), dtype=float)

    def get_cache(self):
        return 123
    
#=============================================================================

class FeatureExtractorNetworkParams(FeatureExtractorParams):
    def __init__(self, network = None, layer = 'softmax', \
                        cache_features = True):
        self.layer = layer
        self.net = network
        self.cache_features = cache_features
    
    def get_id_desc(self):
        return 'name:{0}-layer:{1}'.format( \
                  self.net.__class__, self.layer)

class FeatureExtractorNetwork(FeatureExtractor):
    """
    Extract features using a Network object.
    
    The cached features are saved in a dictionary field of key
    FeatureExtractorNetworkParams.get_id_desc(), which is a dictionary
    with the following fields:
      featdata: ndarray of size [num_bboxes, num_features]
      featidx: {bbox_key -> idx}
          where bbox_key is a string 'xmin-ymin-xmax-ymax' and idx
          refers to the idx in featdata

    NOTE: for efficiency reason and simplicitly, the current implementation
          allows only one type of network during the entire life of
          this class.          
    """
    
    # network to use during the life of any FeatureExtractorCaffe object
    network_ = None

    def __init__(self, anno_image, params):
        """
        *** PRIVATE CONSTRUCTOR *** 
        
        Input: AnnotatedImage and FeatureExtractorNetworkParams
        """
        from annotatedimage import * # HACK: known circular import problem 
        assert isinstance(anno_image, AnnotatedImage)
        assert isinstance(params, FeatureExtractorNetworkParams)
        self.name = 'FeatureExtractorNetwork'
        self.anno_image = anno_image
        self.img = anno_image.get_image() # just for efficiency
        assert self.img.shape[0] == self.anno_image.image_height
        assert self.img.shape[1] == self.anno_image.image_width
        self.params = params
        if FeatureExtractorNetwork.network_:
            assert FeatureExtractorNetwork.network_ == self.params.net, \
                'Only a single network is allowed during the life of '\
                'FeatureExtractorNetwork'
        else:
            FeatureExtractorNetwork.network_ = self.params.net
        # inizialize the cache
        modulename = self.name
        name = self.params.get_id_desc()
        if modulename in self.anno_image.features:
            self.cache = self.anno_image.features[modulename]
        else:
            self.cache = {} 
        if name not in self.cache:
            self.cache[name] = {}
            self.cache[name]['featdata'] = None
            self.cache[name]['featidx'] = {}
    
    def __getstate__(self):
        d = dict(self.__dict__)
        del d['params']
        return d
            
    def extract(self, bboxes):
        # for each bbox:
        width = self.anno_image.image_width
        height = self.anno_image.image_height
        feats = None
        for idx_bbox, bbox in enumerate(bboxes):
            # convert the bbox to absolute, integer values
            bb = bbox.copy().rescale_to_outer_box(width, height)
            bb.convert_coordinates_to_integers()
            modulename = self.name
            name = self.params.get_id_desc()
            key = '{0}-{1}-{2}-{3}'.format(bb.xmin, bb.ymin, bb.xmax, bb.ymax)
            try:
                # try to see if in the cache there are the features we want
                netfeat = self.cache[name]
                feat = netfeat['featdata'][netfeat['featidx'][key], :]
            except:
                # the features are not present :-( we extract them
                img = self.img.copy()
                img = img[bb.ymin:bb.ymax, bb.xmin:bb.xmax]
                feat = FeatureExtractorNetwork.network_.evaluate( \
                              img, layer_name=self.params.layer)
                feat = np.atleast_2d(feat)
                if feat.shape[0] > 1:
                    feat = feat.T  # feat must be a horizontal vector
                assert feat.shape[0] == 1
                # save the feature in the cache, if requested
                if self.params.cache_features:
                    if self.cache[name]['featdata'] == None:
                        self.cache[name]['featdata'] = feat                    
                    else:
                        self.cache[name]['featdata'] = \
                               np.vstack([self.cache[name]['featdata'], feat])
                    self.cache[name]['featidx'][key] = \
                               self.cache[name]['featdata'].shape[0] - 1
            # copy the features
            if feats == None:
                feats = np.ndarray(shape=(len(bboxes), feat.size), dtype=float)
            feats[idx_bbox, :] = feat.copy()
        # return
        assert feats.shape[0] == len(bboxes)
        assert feats.shape[1] > 0
        return feats

    def get_cache(self):
        return self.cache
    



