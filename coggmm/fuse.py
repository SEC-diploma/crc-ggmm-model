"""Complementary concordance of RNA-seq and 16S (both noisy).

16S presence is *not* a gold-standard label. It is a second incomplete assay.
The four classes below are the scientific object of the dual-assay paper:

  dual-supported : 16S+ and high RNA evidence
  rna-rescued    : 16S− and high RNA evidence   (putative 16S false negatives)
  16s-only       : 16S+ and low RNA evidence    (DNA-present / RNA-silent or RNA FN)
  dual-low       : 16S− and low RNA evidence    (likely artefacts)

RNA evidence is tooladapt or prevalence. Thresholds are cohort-internal quantiles
so the partition does not leak a fitted 16S classifier.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dual_supported_mask(y_16s: np.ndarray, rna_score: np.ndarray, q: float = 0.75) -> np.ndarray:
    y = np.asarray(y_16s, int)
    s = np.asarray(rna_score, float)
    thr = float(np.nanquantile(s, q))
    return (y == 1) & (s >= thr)


def concordance_classes(
    y_16s: np.ndarray,
    rna_score: np.ndarray,
    q_high: float = 0.75,
) -> pd.DataFrame:
    y = np.asarray(y_16s, int)
    s = np.asarray(rna_score, float)
    thr = float(np.nanquantile(s[np.isfinite(s)], q_high))
    high = s >= thr
    cls = np.full(y.shape, "dual-low", dtype=object)
    cls[high & (y == 1)] = "dual-supported"
    cls[high & (y == 0)] = "rna-rescued"
    cls[(~high) & (y == 1)] = "16s-only"
    cls[(~high) & (y == 0)] = "dual-low"
    return pd.DataFrame({"y_16s": y, "rna_score": s, "rna_high": high, "class": cls})
