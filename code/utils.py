import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def _fast_hist(label_true, label_pred, n_class):
    hist = np.bincount(n_class * label_true.astype(np.int) + label_pred, minlength=n_class**2).reshape(n_class, n_class)
    return hist

class Measurement(object):
    def average_precision(self, scores, targets):
        return average_precision_score(targets, scores)

    def auc_score(self, scores, targets):
        return roc_auc_score(targets, scores)

    def calculate_conf_mat(self, scores, targets, threshold):
        #return a confusion matrix of which lines represents ture labels and columns represents predicted labels.
        preds = np.zeros(scores.shape, dtype=np.int)
        preds[scores>threshold] = 1
        preds[scores<=threshold] = 0

        hist = _fast_hist(targets, preds, n_class=2)
        return hist

    def metrics(self, conf_mat):
        e = 1e-15
        conf_mat = conf_mat.astype(np.float64)
        success = np.diag(conf_mat)
        acc = success.sum() / conf_mat.sum()
        recall = success / conf_mat.sum(axis=1)
        precision = success / conf_mat.sum(axis=0)
        specificity = recall[0]
        recall = recall[1]
        precision = precision[1]
        f1 = precision * recall * 2 / (precision + recall + e)
        return acc, precision, recall, specificity, f1

class Evaluater(Measurement):
    def evaluate(self, scores, targets, threshold):
        try:
            ap = self.average_precision(scores, targets)
        except:
            ap = -200000
        try:
            auc = self.auc_score(scores, targets)
        except:
            auc=-10000
        conf_mat = self.calculate_conf_mat(scores, targets, threshold)
        acc, precision, recall, specificity, f1 = self.metrics(conf_mat)
        return ap, auc, acc, precision, recall, specificity, f1, conf_mat

