#!/usr/bin/env python3
"""Synthetic CoGGMM demo (no 200 GB r² matrices required).

Generates a Gamma background + Gaussian bump, recovers mixture weights,
and prints AUROC for prevalence vs fraction vs posterior mean.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coggmm import (  # noqa: E402
    GammaGaussianMix,
    SingleGamma,
    estimate_wg_em,
    fit_gamma_gaussian_mixture,
    fit_single_gamma_from_moments,
    pnxfraction_from_r2,
    prevalence,
)


def main() -> None:
    rng = np.random.default_rng(7)
    n_pos, n_neg, n_s, n_g = 80, 400, 40, 200
    # Recurrent "true" taxa: high prevalence, r² shifted toward a Gaussian bump
    pos_counts = (rng.random((n_pos, n_s)) < 0.85).astype(float) * rng.poisson(8, (n_pos, n_s))
    neg_counts = (rng.random((n_neg, n_s)) < 0.15).astype(float) * rng.poisson(1, (n_neg, n_s))
    counts = np.vstack([pos_counts, neg_counts])
    y = np.array([1] * n_pos + [0] * n_neg)
    prev = prevalence(counts)

    r2_bg = rng.gamma(0.6, 0.04, size=n_neg * n_g).clip(0, 1)
    r2_fg = np.clip(rng.normal(0.18, 0.06, size=n_pos * n_g), 0, 1)
    g, _ = fit_single_gamma_from_moments(float(r2_bg.mean()), float(r2_bg.var()))
    mix, meta = fit_gamma_gaussian_mixture(r2_fg, g, mode="soft_mu")
    print("Stage2", meta["status"], "mu", round(mix.gauss_mean, 4), "w_N", round(mix.w_gaussian, 3))

    frac = np.array(
        [estimate_wg_em(r2_fg[i * n_g : (i + 1) * n_g], mix)["frac_gaussian"] for i in range(n_pos)]
        + [estimate_wg_em(r2_bg[i * n_g : (i + 1) * n_g], mix)["frac_gaussian"] for i in range(n_neg)]
    )
    pnx = np.array(
        [pnxfraction_from_r2(r2_fg[i * n_g : (i + 1) * n_g], mix) for i in range(n_pos)]
        + [pnxfraction_from_r2(r2_bg[i * n_g : (i + 1) * n_g], mix) for i in range(n_neg)]
    )
    print("AUROC prevalence", round(roc_auc_score(y, prev), 3))
    print("AUROC fraction  ", round(roc_auc_score(y, frac), 3))
    print("AUROC pnx       ", round(roc_auc_score(y, pnx), 3))
    print("On real 8x4 data prevalence still dominates; this demo only checks the API.")


if __name__ == "__main__":
    main()
