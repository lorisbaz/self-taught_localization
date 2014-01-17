

class Heatmap:
    """class for Heatmap"""

    def __init__(self):
        raise NotImplementedError()

    def add_val(self, x, y, val):
        raise NotImplementedError()

    def add_val(self, x, y, width, height, val):
        raise NotImplementedError()

    def normalize_counts(self):
        raise NotImplementedError()

    def get(self):
	"""
	Returns a ndarray.float
	"""
        raise NotImplementedError()

    def export_to_jpeg(self):
	"""
	Returns a string of bytes containing a Jpeg visualization
	"""
        raise NotImplementedError()

    def save_to_jpeg(self, filename):
        raise NotImplementedError()

    @staticmethod
    def average_heatmaps(heatmaps):
	"""
	Returns a Heatmap object produced by averaging a set of
	heatmaps.
	"""


#=============================================================================
