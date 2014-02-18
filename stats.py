import numpy as np

class Stats:
    """
    Class that represents a set of statistics computed for each image.
    Public fields:
    - precision, recall, TP, FP, FN, detection_rate
    """

    def __init__(self):
        self.overlap = []
        self.precision = []
        self.recall = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.detection_rate = []

    def __str__(self):
        return 'Stats {0} - Precision: {1}, Recall: {2}, TP: {3}, FP: {4}, ' \
               'FN: {5}, Detection rate: {6}' \
               .format(self.image_id, self.precision, self.recall, self.TP, \
               self.FP, self.FN, self.detection_rate)

    def compute_stats(pred_bboxes, gt_bboxes, IoU_threshold = 0.5):
        """
        Compute the statistics given a dictionary of BBox objects of the 
        predictions and the ground truth. IoU_threshols is the Intersection 
        over Union threshold given by the PASCAL VOC evaluationa criteria. 
        Note: IoU is the same as Jaccard similarity.
        """
        # Flat pred_bboxes & gt_bboxes
        pred_bboxes_flat, pred_labels_flat = \
                                    self.flat_anno_bboxes_(pred_bboxes)
        gt_bboxes_flat, gt_labels_flat = \
                                    self.flat_anno_bboxes_(gt_bboxes)

        # Sort predictions by accuracy 
        pred_confidence = []
        for i in len(pred_bboxes_flat):
            pred_confidence.append(pred_bboxes_flat[i].confidence)
        idx_sort = np.argsort(pred_confidence)

        # Compute overlap (IoU)
        self.overlap = np.zeros(len(pred_bboxes_flat))
        used_gts = []
        for i in idx_sort:
            IoU = np.zeros(gt_bboxes_flat)
            for j in len(gt_bboxes_flat): 
                if not(j in the used_gts): 
                    IoU[j] = pred_bboxes_flat[i].\
                                jaccard_similarity(gt_bboxes_flat[j])
            jmax = np.argmax(IoU)  
            self.overlap[i] = IoU[jamx]
            
            # TODOO: if conditions for FP and TP
            self.FP += 1 
            self.TP += 1     
            # do not use this GT again (already matched)
            used_gts.append(jmax)
        
        # Compute FN as (tot num positive - TP)
        self.FN = len(gt_bboxes_flat) - self.TP
        # Recall

        # Precition

    def flat_anno_bboxes_(bboxes):
        bboxes_flat = []
        labels_flat = []
        for label in bboxes.keys():
            bboxes_flat.extend(bboxes[label].bboxes)
            rep_lab = []
            for i in len(bboxes[label].bboxes):
                rep_lab.append(label)
            labels_flat.extend(rep_lab)
        return bboxes_flat, labels_flat

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


