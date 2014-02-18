import numpy as np

class Stats:
    """
    Class that represents a set of statistics computed for each image.
    Public fields:
    - precision, recall, TP, FP, FN, detection_rate
    """

    def __init__(self):
        self.overlap = []
        self.precision = 0
        self.recall = 0
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.detection_rate = 0

    def __str__(self):
        return 'Stats {0} - Precision: {1}, Recall: {2}, TP: {3}, FP: {4}, ' \
               'FN: {5}, Detection rate: {6}' \
               .format(self.image_id, self.precision, self.recall, self.TP, \
               self.FP, self.FN, self.detection_rate)

    def compute_stats(pred_bboxes, gt_bboxes, IoU_threshold = 0.5):
        """
        Compute the statistics given a list of BBox objects of the predictions
        and the ground truth. IoU_threshols is the Intersection over Union
        threshold given by the PASCAL VOC evaluationa criteria. Note: IoU is 
        the same as Jaccard similarity.
        """
        # TODO: compute different statistics here


    @staticmethod
    def average_results(self):
        """
        Returns a Stats object that is the averaged statistics and the 
        histogram of overlapped regions.
        """
        stats_avg = Stats()
        hist_overlap = np.zeros(n_bins)
        # TODO: compute averages and histograms

        return stats_avg, hist_overlap

