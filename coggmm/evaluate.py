"""Evaluation utilities. 16S intersection is a *noisy validator*, not a gold standard."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _mask(y, s):
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s) & np.isfinite(y)
    return y[m], s[m]


def auroc(y, s) -> float:
    y, s = _mask(y, s)
    if y.size < 8 or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, s))


def auprc(y, s) -> float:
    y, s = _mask(y, s)
    if y.size < 8 or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, s))


def roc_points(y, s):
    y, s = _mask(y, s)
    if y.size < 8 or len(np.unique(y)) < 2:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])
    fpr, tpr, _ = roc_curve(y, s)
    return fpr, tpr


def precision_recall_at_k(y, s, k: int, taxid=None):
    """Top-K precision/recall. Ties broken by (-score, taxid) for reproducibility.

    Returns (precision, recall, n_pos_in_top, k_eff).
    """
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s) & np.isfinite(y)
    y, s = y[m], s[m]
    if y.size == 0 or int(y.sum()) == 0:
        return float("nan"), float("nan"), 0, 0
    k_eff = int(min(k, y.size))
    if taxid is None:
        taxid = np.arange(y.size)
    else:
        taxid = np.asarray(taxid)[m]
    order = np.lexsort((taxid, -s))
    top = y[order[:k_eff]]
    n_hit = int(top.sum())
    prec = n_hit / k_eff
    rec = n_hit / float(y.sum())
    return float(prec), float(rec), n_hit, k_eff
