import numpy as np

class Stats:
    """
    Class that represents a set of statistics computed for each image.
    Public fields:
    - precision, recall, TP, FP, number of positives, detection_rate
    """

    def __init__(self):
        self.overlap = []
        self.TP = []
        self.FP = []
        self.NPOS = 0 
        self.precision = []
        self.recall = []
        self.detection_rate = 0.0
        self.average_prec = 0.0

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

        # Compute overlap (IoU) -> code translated from PASCAL VOC
        self.overlap = np.zeros(len(pred_bboxes_flat))
        gt_det = np.zeros(len(gt_bboxes_flat), 'bool')
        self.FP = np.zeros(len(pred_bboxes_flat))  
        self.TP = np.zeros(len(pred_bboxes_flat)) 
        self.overlap = np.zeros(len(pred_bboxes_flat)) 
        for i in idx_sort:
            ovmax = float("-inf")
            for j in len(gt_bboxes_flat):  
                ov = pred_bboxes_flat[i].jaccard_similarity(gt_bboxes_flat[j])
                if ov>ovmax:
                    ovmax = ov
                    jamx = j
            # check if it is FP or TP        
            self.overlap[i] = ovmax
            if not(ovmax>=IoU_threshold):
                if not(gt_det[jmax]):
                    self.TP[i] = 1 # true positive
                    gt_det[jmax] = True # association done!
                else:
                    self.FP[i] = 1 # false positive (multiple detection)        
            else:
                self.FP[i] = 1 # false positive
        # Store the tot num positive for the actual image 
        self.NPOS = len(gt_bboxes_flat)       

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
    def aggregate_results(self, stats_list, n_bins):
        """
        Returns a Stats object that contains the aggregated statistics and the 
        histogram of overlapped regions.
        """
        hist_overlap = np.zeros(n_bins)
        stats_aggr = Stats() # aggregate stats
        # Aggregate data
        for i in range(len(stats_vec)):
            stats_aggr.TP.extend(stats_list[i].TP)
            stats_aggr.FP.extend(stats_list[i].FP) 
            stats_aggr.overlap.extend(stats_list[i].overlap)
            stats_aggr.NPOS += stats_list[i].NPOS
        # Cumulative precision/recall (PASCAL stuff)    
        stats_aggr.precision = np.cumsum(stats_aggr.TP)/float(stats_aggr.NPOS)
        stats_aggr.recall = np.cumsum(stats_aggr.TP)/(np.cumsum(stats_aggr.TP)\
                                    - np.cumsum(stats_aggr.FP)) 
        # Compute the average precision        
        stats_aggr.average_prec = 0.0
        for t in np.linspace(0,1,11):
            ps = stats_aggr.precision(stats_aggr.recall>=t)
            if np.shape(ps)[0]==0:
                p = 0
            else:
                p = np.max(ps)
            stats_aggr.average_prec += p/11
        # Compute the detection rate
        stats_aggr.detection_rate = np.sum(stats_aggr.TP)/float(stats_aggr.NPOS)
        # Create the histogram of overlap
        hist_overlap = np.histogram(stats_aggr.overlap, n_bins)
        
        return stats_avg, hist_overlap


