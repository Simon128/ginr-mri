"""
as described here:
https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
and here:
https://arxiv.org/pdf/1905.05667.pdf
"""

import numpy as np
import math
from scipy.special import binom

def compute_purity(cluster_predictions, labels, num_clusters):
    cluster_predictions = np.array(cluster_predictions)
    labels = np.array(labels)
    maj_cluster_labels = get_majority_cluster_labels(cluster_predictions, labels, num_clusters)
    N = len(cluster_predictions)
    purity = 0

    for n in range(num_clusters):
        mask = cluster_predictions == n
        tp = labels[mask] == maj_cluster_labels[n]
        purity += sum(tp)

    return purity / N

def precision(cluster_predictions, labels, num_clusters, tpfp = None):
    if tpfp is None:
        tp, _, fp, _ = tptnfpfn(cluster_predictions, labels, num_clusters)
    else:
        tp, fp = tpfp
    return tp / (tp + fp)

def recall(cluster_predictions, labels, num_clusters, tpfn = None):
    if tpfn is None:
        tp, _, _, fn = tptnfpfn(cluster_predictions, labels, num_clusters)
    else:
        tp, fn = tpfn
    return tp / (tp + fn)

def jaccard_coefficient(cluster_predictions, labels, num_clusters, tpfpfn = None):
    if tpfpfn is None:
        tp, _, fp, fn = tptnfpfn(cluster_predictions, labels, num_clusters)
    else:
        tp, fp, fn = tpfpfn
    return tp / (tp + fp + fn)

def rand_coefficient(cluster_predictions, labels, num_clusters, _tptnfpfn = None):
    if _tptnfpfn is None:
        tp, tn, fp, fn = tptnfpfn(cluster_predictions, labels, num_clusters)
    else:
        tp, tn, fp, fn = _tptnfpfn
    return (tp + tn) / (tp + tn + fp + fn)

def f_beta_precision_recall(cluster_predictions, labels, num_clusters, beta=1, tpfpfn = None):
    if tpfpfn is None:
        tp, _, fp, fn = tptnfpfn(cluster_predictions, labels, num_clusters)
    else: 
        tp, fp, fn = tpfpfn
    p = precision(cluster_predictions, labels, num_clusters, (tp, fp)) 
    r = recall(cluster_predictions, labels, num_clusters, (tp, fn))
    beta_sq = math.pow(beta, 2)
    return ((beta_sq + 1) * p * r) / (beta_sq * p + r), p, r

def tptnfpfn(cluster_predictions, labels, num_clusters):
    tp = 0
    tpfp = 0
    fp = 0
    fn = 0
    num_labels = len(np.unique(labels))
    cluster_predictions = np.array(cluster_predictions)
    labels = np.array(labels)

    label_counts_per_cluster = []
    for n in range(num_clusters):
        mask = cluster_predictions == n
        pred_labels = labels[mask]
        counts = [sum(pred_labels == l) for l in range(num_labels)]
        label_counts_per_cluster.append(counts)
        for c in counts:
            if c >= 2:
                tp += binom(c, 2)
        tpfp += binom(sum(mask), 2)

    for label in range(num_labels):
        for cluster in range(num_clusters):
            assigned = label_counts_per_cluster[cluster][label]
            missassigned = sum([label_counts_per_cluster[c][label] for c in range(cluster + 1, num_clusters)])
            fn += assigned * missassigned

    fp = tpfp - tp
    total_negatives = binom(len(cluster_predictions), 2) - tpfp 
    tn = total_negatives - fn
    return tp, tn, fp, float(fn)

def get_majority_cluster_labels(cluster_predictions, labels, num_clusters):
    maj_labels = []
    for n in range(num_clusters):
        mask = cluster_predictions == n
        pred_labels = labels[mask]
        values, counts = np.unique(pred_labels, return_counts=True)
        ind = np.argmax(counts)
        maj_labels.append(values[ind])
    return maj_labels

if __name__ == "__main__":
    # testing example from: https://nlp.stanford.edu/IR-book/html/htmledition/evaluation-of-clustering-1.html
    cluster_predictions = [0] * 6 + [1] * 6 + [2] * 5
    #x = 0, o = 1, raute = 2
    labels = [0] * 5 + [1, 0] + [1] * 4 + [2, 0, 0] + [2] * 3
    num_clusters = 3
    tp, tn, fp, fn = tptnfpfn(cluster_predictions, labels, num_clusters)
    f_beta, p, r = f_beta_precision_recall(cluster_predictions, labels, num_clusters, beta=1)
    purity = compute_purity(cluster_predictions, labels, num_clusters)
    test  = 5
