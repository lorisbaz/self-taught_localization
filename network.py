

class Network:
    """Network class"""

    def __init__(self):
        raise NotImplementedError()

    def eval(self, image, layer_name = 'softmax'):
	"""Evaluate the network given a an image, and export the layer
	named 'layer_name'

	image: ndarray.uint8
	layer_name: string
	"""
        raise NotImplementedError()

    def get_label_id(self, label):
	"""
	Returns the unique label id \in {0, ..., num_labels-1}
	"""
	pass

    def get_label_desc(self, label):
	"""
	Returns a textual description of the label
	"""
	pass

    def get_labels(self):
	"""
	Returns the list of labels (in order of id) that the network can use
	"""
	pass

#=============================================================================

class NetworkDecaf(Network):
    """
    """
    def __init__(self, model_spec_filename, model_filename):
	# here there is the loading of the model
	pass

    def eval(self, image, layer_name = 'softmax'):
	"""Evaluate the network given a an image, and export the layer
	"""


#=============================================================================

# TODO here to implement Caffe
