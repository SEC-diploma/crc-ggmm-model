"""Joint library-size normalisation used before Pearson / PC-partial r²."""
from __future__ import annotations

import numpy as np


def joint_logcpm(host: np.ndarray, micro: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """host (n_genes, n), micro (n_taxa, n) raw counts → log2(CPM+1)."""
    H = np.asarray(host, float)
    M = np.asarray(micro, float)
    lib = H.sum(axis=0) + M.sum(axis=0)
    lib = np.maximum(lib, 1.0)
    Hn = np.log2(H / lib * 1e6 + 1.0)
    Mn = np.log2(M / lib * 1e6 + 1.0)
    return Hn, Mn
