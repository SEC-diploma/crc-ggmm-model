#!/usr/bin/env python3
"""INVADEseq OSCC adapter: barcode overlap + GMM/Gamma medians.

Primary samples OSCC_11–15; OSCC_16–17 are secondary (weak 16S).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def barcode_overlap(host: pd.Series, rna: pd.Series, s16: pd.Series) -> dict:
    h, r, s = set(host.index[host > 0]), set(rna.index[rna > 0]), set(s16.index[s16 > 0])
    triple = h & r & s
    return {
        "n_host": len(h),
        "n_rna": len(r),
        "n_16s": len(s),
        "n_host_16s": len(h & s),
        "n_triple": len(triple),
        "pct_triple_of_host": 100.0 * len(triple) / max(len(h), 1),
    }


def summarise_gmm_table(path: Path) -> pd.Series:
    df = pd.read_csv(path, sep="\t")
    col_g = next(c for c in df.columns if "gmm" in c.lower() or "true" in c.lower())
    col_n = next((c for c in df.columns if "gamma" in c.lower()), None)
    out = {"gmm_median": float(df[col_g].median())}
    if col_n:
        out["gamma_median"] = float(df[col_n].median())
    return pd.Series(out)


if __name__ == "__main__":
    print("See figures/Fig7_singlecell_invade.png for the locked OSCC numbers.")
