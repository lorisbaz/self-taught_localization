from util import *
import logging
import numpy as np

class Stats:
    """
    Class that represents a set of statistics computed for each image.
    Public fields:
    - confidence: confidence of each predicted bbox
    - overlap: bboxes overlap vector accordingly to the PASCAL VOC criterion
    - overlap_for_ABO: bboxes overlap matrix for all the predicted bboxes and 
        GTs. It does not exclude the GT once associated to a prediction.
    - maxoverlap_for_ABO: max over predicted bboxes (useful during the 
        aggregation)    
    - TP, FP: tru positive and false positive vectors
    - NPOS: number of GT bboxes
    - precision, recall: classic precision and recall (PASCAL VOC criterion)
    - average_precision: break point precision-recall curve (PASCAL VOC)
    - ABO: averate best overlap - defined as sum over GT of the max overlap
        with pred objects
    - detection_rate: NOT USED
    """

    def __init__(self):
        self.confidence = []
        self.overlap = []
        self.overlap_for_ABO = []
        self.maxoverlap_for_ABO = []
        self.TP = []
        self.FP = []
        self.NPOS = 0 
        self.precision = []
        self.recall = []
        self.ABO = []
        self.average_prec = 0.0
        self.detection_rate = 0.0

    def __str__(self):
        return 'Stats - Overlap: {0}, TP: {1}, FP: {2}, N pos: {3}, ' \
               'Precision: {4}, Recall: {5}, Detection rate: {6}, ' \
               'Confidence: {7}, AVG precision: {8}'\
               .format(self.overlap, self.TP, self.FP, self.NPOS, \
                       self.precision, self.recall, self.detection_rate, \
                       self.confidence, self.average_prec)

    def compute_stats(self, pred_bboxes, gt_bboxes, IoU_threshold = 0.5, \
                      fp_overlap_zero = False, max_subwin = 0):
        """
        Compute the statistics given a list of BBox objects of the 
        predictions and the ground truth. IoU_threshols is the Intersection 
        over Union threshold given by the PASCAL VOC evaluationa criteria. 
        Note: IoU is the same as Jaccard similarity.
        """
        # Sort predictions by accuracy 
        pred_confidence = []
        for i in range(len(pred_bboxes)):
            pred_confidence.append(pred_bboxes[i].confidence)
        idx_sort = np.argsort(pred_confidence)[::-1]
        self.confidence = pred_confidence 
        # Compute overlap (IoU) -> code translated from PASCAL VOC
        gt_det = np.zeros(len(gt_bboxes), 'bool')
        self.FP = np.zeros(len(idx_sort))  
        self.TP = np.zeros(len(idx_sort)) 
        self.overlap = np.zeros(len(idx_sort))
        for i in idx_sort:
            ovmax = 0.0
            for j in range(len(gt_bboxes)):
                ov = pred_bboxes[i].jaccard_similarity(gt_bboxes[j])
                #print '{0}'.format(ov)
                if ov>ovmax:
                    ovmax = ov
                    jmax = j
            # check if it is FP or TP        
            self.overlap[i] = ovmax
            if ovmax>=IoU_threshold:
                if not(gt_det[jmax]):
                    self.TP[i] = 1 # true positive
                    gt_det[jmax] = True # association done!
                else:
                    self.FP[i] = 1 # false positive (multiple detection) 
                    if fp_overlap_zero:
                        self.overlap[i] = 0.0
            else:
                self.FP[i] = 1 # false positive
        # Compute overlap for ABO (does not exclude pred_objects)
        self.overlap_for_ABO = np.zeros((len(gt_bboxes), len(idx_sort)))
        for j in range(len(gt_bboxes)):
            for i in idx_sort:
                self.overlap_for_ABO[j,i] = \
                        pred_bboxes[i].jaccard_similarity(gt_bboxes[j])
        # Store the tot num positive for the actual image 
        self.NPOS = len(gt_bboxes) 
        # Keep the top num_windows 
        if max_subwin > 0:
            # select num_windows indexes
            idx_sort = idx_sort[0:min(len(idx_sort), max_subwin)] 
            self.overlap = self.overlap[idx_sort]
            self.FP = self.FP[idx_sort]
            self.TP = self.TP[idx_sort]
            self.confidence = np.array(self.confidence)[idx_sort].tolist()
            # it is a list to keep compatibility

    @staticmethod
    def flat_anno_bboxes(bboxes):
        bboxes_flat = []
        labels_flat = []
        for label in bboxes.keys():
            bboxes_flat.extend(bboxes[label].bboxes)
            rep_lab = []
            for i in range(len(bboxes[label].bboxes)):
                rep_lab.append(label)
            labels_flat.extend(rep_lab)
        return bboxes_flat, labels_flat

    @staticmethod
    def aggregate_results(stats_list, n_bins=10, topN = float("inf")):
        """
        Returns a Stats object that contains the aggregated statistics and the 
        histogram of overlapped regions.
        """
        hist_overlap = np.zeros(n_bins)
        stats_aggr = Stats() # aggregate stats
        # Aggregate data (selecting topN for each stats obj)
        for i in range(len(stats_list)):
            # select topN
            stats_now = Stats()
            idx_sort = np.argsort(stats_list[i].confidence)[::-1]
            if len(idx_sort)==0:
                logging.warning('The stats {0} is empty!'.format(i))
                continue
            if len(idx_sort)>topN:
                idx_sort = idx_sort[0:topN]
            stats_now.confidence = np.array(stats_list[i].confidence)[idx_sort]
            stats_now.TP = np.array(stats_list[i].TP)[idx_sort]
            stats_now.FP = np.array(stats_list[i].FP)[idx_sort]
            stats_now.overlap = np.array(stats_list[i].overlap)[idx_sort]
            stats_now.maxoverlap_for_ABO = np.max(np.array(stats_list[i].\
                                        overlap_for_ABO)[:, idx_sort], axis=1)
            stats_now.NPOS = stats_list[i].NPOS
            # aggregate 
            stats_aggr.confidence.extend(stats_now.confidence)
            stats_aggr.TP.extend(stats_now.TP)
            stats_aggr.FP.extend(stats_now.FP) 
            stats_aggr.overlap.extend(stats_now.overlap)
            stats_aggr.maxoverlap_for_ABO.extend(stats_now.maxoverlap_for_ABO)
            stats_aggr.NPOS += stats_now.NPOS
        # Sort by confidence
        idx_sort = np.argsort(stats_aggr.confidence)[::-1]
        stats_aggr.confidence = np.array(stats_aggr.confidence)[idx_sort]
        stats_aggr.TP = np.array(stats_aggr.TP)[idx_sort]
        stats_aggr.FP = np.array(stats_aggr.FP)[idx_sort]
        stats_aggr.overlap = np.array(stats_aggr.overlap)[idx_sort] 
        # Cumulative precision/recall (PASCAL stuff)    
        stats_aggr.recall = np.cumsum(stats_aggr.TP)/float(stats_aggr.NPOS)
        stats_aggr.precision = np.cumsum(stats_aggr.TP) / \
            np.float32((np.cumsum(stats_aggr.TP) + np.cumsum(stats_aggr.FP)))
        # Compute the average precision        
        stats_aggr.average_prec = 0.0
        for t in np.linspace(0,1,11):
            ps = stats_aggr.precision[stats_aggr.recall >= t]
            if np.shape(ps)[0]==0:
                p = 0.0
            else:
                p = float(np.max(ps))
            stats_aggr.average_prec += p / 11.0
        # Compute ABO
        stats_aggr.ABO = np.sum(stats_aggr.maxoverlap_for_ABO)/stats_aggr.NPOS
        # Compute the detection rate
        stats_aggr.detection_rate = np.sum(stats_aggr.TP)/float(stats_aggr.NPOS)
        # Create the histogram of overlap
        hist_overlap = np.histogram(stats_aggr.overlap, \
                                bins = n_bins, range = (0,1))
        return stats_aggr, hist_overlap

