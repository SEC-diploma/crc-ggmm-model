"""TaxID-level scores: prevalence, GGMM fraction, Bayesian posterior, tooladapt.

None of these functions takes a 16S label vector. 16S is not used at scoring time.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import gamma as gamma_dist
from scipy.stats import norm

from .mixture import W_CLIP, GammaGaussianMix, estimate_wg_em

LEX_SCALE = 0.25


def prevalence(counts: np.ndarray) -> np.ndarray:
    """Detection frequency across samples. counts: (n_taxa, n_samples)."""
    counts = np.asarray(counts, float)
    return np.mean(counts > 0, axis=1)


def fraction_from_r2(r2_row: np.ndarray, mix: GammaGaussianMix, min_pairs: int = 50) -> float:
    """Stage-3 EM Gaussian weight for one TaxID (GGMM fraction)."""
    out = estimate_wg_em(np.asarray(r2_row, float), mix, min_pairs=min_pairs)
    frac = float(out["frac_gaussian"])
    if mix.gauss_mean < mix.gamma.mean and np.isfinite(frac):
        frac = 1.0 - frac
    return float(np.clip(frac, W_CLIP[0], W_CLIP[1])) if np.isfinite(frac) else float("nan")


def bayes_posterior(x: np.ndarray, mix: GammaGaussianMix) -> np.ndarray:
    """Pointwise P(N | x) under the fitted mixture, global π = w_gamma frozen."""
    x = np.maximum(np.asarray(x, float), 1e-12)
    f_g = gamma_dist.pdf(x, a=mix.gamma.shape, scale=mix.gamma.scale)
    f_n = norm.pdf(x, mix.gauss_mean, mix.gauss_std)
    num = mix.w_gaussian * f_n
    den = mix.w_gamma * f_g + num + 1e-300
    p = num / den
    if mix.gauss_mean < mix.gamma.mean:
        p = 1.0 - p
    return np.clip(p, 0.0, 1.0)


def pnxfraction_from_r2(r2_row: np.ndarray, mix: GammaGaussianMix) -> float:
    """Mean Bayesian posterior P(N | r²) across genes (pnxfraction)."""
    x = np.asarray(r2_row, float)
    x = x[np.isfinite(x) & (x >= 0.0) & (x <= 1.0)]
    if x.size < 50:
        return float("nan")
    return float(np.mean(bayes_posterior(x, mix)))


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    m = np.isfinite(x)
    out = np.full_like(x, np.nan, dtype=float)
    if m.sum() < 2:
        return out
    sd = float(np.std(x[m]))
    if sd < 1e-12:
        out[m] = 0.0
        return out
    out[m] = (x[m] - float(np.mean(x[m]))) / sd
    return out


def lex_minus(prev: np.ndarray, pnx: np.ndarray, scale: float = LEX_SCALE) -> np.ndarray:
    """Dictionary-order tie-break: keep prevalence ranks, subtract residual pnx.

    |0.25 Δ z| < Δ so adjacent prevalence bins cannot swap.
    Residual pnx vs 16S intersection is weakly *anti*-correlated, hence minus.
    """
    prev = np.asarray(prev, float)
    pnx = np.asarray(pnx, float)
    m = np.isfinite(prev) & np.isfinite(pnx)
    out = np.array(prev, copy=True)
    if m.sum() < 8:
        return out
    p = prev[m]
    q = pnx[m]
    X = np.column_stack([np.ones(p.size), p])
    beta, *_ = np.linalg.lstsq(X, q, rcond=None)
    resid = q - (beta[0] + beta[1] * p)
    uniq = np.unique(np.round(p, 12))
    if uniq.size < 2:
        delta = 1.0 / max(p.size, 1)
    else:
        delta = float(np.min(np.diff(np.sort(uniq))))
    z = np.clip(_zscore(resid), -3.0, 3.0)
    out[m] = p - scale * delta * z
    return out


def tooladapt_score(
    prev_by_tool: dict[str, np.ndarray],
    pnx: np.ndarray,
    current_tool: str,
) -> np.ndarray:
    """Kraken uses four-tool mean prevalence; aligners keep their own prevalence.

    Then apply lex_minus(pnx) as a within-bin tie-break.
    """
    stacked = np.vstack([prev_by_tool[t] for t in prev_by_tool])
    prev_mean4 = np.nanmean(stacked, axis=0)
    own = prev_by_tool[current_tool]
    primary = prev_mean4 if current_tool.lower() == "kraken" else own
    return lex_minus(primary, pnx)
