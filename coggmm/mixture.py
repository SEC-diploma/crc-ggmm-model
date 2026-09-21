#!/usr/bin/env python3
"""Core Γ + Normal mixture utilities for August-work two-stage pipeline.

Model:
  f(r²) = w1 * Gamma(r²; k, θ) + w2 * Normal(r²; μ, σ),  w1 + w2 = 1

Design (confirmed):
  Stage1: fit single Gamma on ALL outside r² (no subsample)
  Stage2: fix Gamma; fit w, μ, σ on ALL intersection r² (unconstrained)
  Stage3: fix k,θ,μ,σ; per-taxid EM for w_g only; clip frac_gaussian to [0.01, 0.99]
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import gamma, norm

W_CLIP = (0.01, 0.99)


@dataclass
class SingleGamma:
    shape: float
    scale: float

    @property
    def mean(self) -> float:
        return self.shape * self.scale

    def pdf(self, xs: np.ndarray) -> np.ndarray:
        xs_eval = np.maximum(xs, 1e-12)
        return np.nan_to_num(gamma.pdf(xs_eval, a=self.shape, scale=self.scale), nan=0.0, posinf=0.0)

    def pointwise(self, x: np.ndarray) -> np.ndarray:
        return self.pdf(x)


@dataclass
class GammaGaussianMix:
    gamma: SingleGamma
    w_gamma: float
    gauss_mean: float
    gauss_std: float

    @property
    def w_gaussian(self) -> float:
        return 1.0 - self.w_gamma

    def pdf(self, xs: np.ndarray) -> np.ndarray:
        p_g, p_n = self.component_pdfs(xs)
        return p_g + p_n

    def component_pdfs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        p_gamma = self.w_gamma * self.gamma.pointwise(x)
        p_gauss = self.w_gaussian * norm.pdf(x, self.gauss_mean, self.gauss_std)
        return p_gamma, p_gauss


def iter_r2_rows(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        fh.readline()
        for line in fh:
            taxid, sep, rest = line.partition("\t")
            if not sep:
                continue
            vals = np.fromstring(rest, sep="\t", dtype=np.float64)
            m = np.isfinite(vals) & (vals >= 0.0) & (vals <= 1.0)
            yield taxid.strip(), vals[m]


def stream_r2_all(path: Path, label: str = "", progress_every: int = 200) -> tuple[np.ndarray, int]:
    """Load ALL valid r² into one array. Returns (values, n_taxa)."""
    chunks: list[np.ndarray] = []
    n_taxa = 0
    for taxid, r2 in iter_r2_rows(path):
        n_taxa += 1
        if r2.size:
            chunks.append(r2)
        if label and n_taxa % progress_every == 0:
            n_so_far = sum(c.size for c in chunks)
            print(f"    [{label}] taxa={n_taxa:,} n_r2={n_so_far:,}", flush=True)
    out = np.concatenate(chunks) if chunks else np.array([], dtype=np.float64)
    if label:
        print(f"    [{label}] DONE taxa={n_taxa:,} n_r2={out.size:,}", flush=True)
    return out, n_taxa


def stream_moments(path: Path, label: str = "") -> tuple[float, float, int, int]:
    """One-pass mean / E[x²] over ALL r² without keeping the full array."""
    n = 0
    n_taxa = 0
    sum_x = 0.0
    sum_x2 = 0.0
    for _, r2 in iter_r2_rows(path):
        n_taxa += 1
        if not r2.size:
            continue
        n += int(r2.size)
        sum_x += float(r2.sum())
        sum_x2 += float(np.dot(r2, r2))
        if label and n_taxa % 200 == 0:
            print(f"    [{label}] taxa={n_taxa:,} n_r2={n:,}", flush=True)
    if n < 50:
        return float("nan"), float("nan"), n, n_taxa
    mean = sum_x / n
    # population second moment -> variance with /n (same as np.var default ddof=0)
    var = max(sum_x2 / n - mean * mean, 1e-8)
    if label:
        print(f"    [{label}] DONE taxa={n_taxa:,} n_r2={n:,} mean={mean:.6g} var={var:.6g}", flush=True)
    return mean, var, n, n_taxa


def fit_single_gamma_from_moments(
    mean: float,
    var: float,
    *,
    shape_min: float = 0.05,
    shape_max: float = 50.0,
) -> tuple[SingleGamma, str]:
    """Moment match; clip shape then set scale=mean/shape to preserve data mean (Q9a)."""
    if not np.isfinite(mean) or not np.isfinite(var) or mean <= 0:
        return SingleGamma(float("nan"), float("nan")), "bad_moments"
    scale0 = max(var / mean, 1e-5)
    shape0 = mean / scale0
    shape = float(np.clip(shape0, shape_min, shape_max))
    # preserve empirical mean after clip
    scale = float(np.clip(mean / shape, 1e-5, 1.0))
    return SingleGamma(shape, scale), "ok"


def fit_single_gamma_full(
    path: Path,
    label: str = "outside",
    *,
    shape_min: float = 0.05,
) -> tuple[SingleGamma, dict]:
    """Stage1: single Gamma on ALL outside r² via streaming moments (no subsample)."""
    mean, var, n, n_taxa = stream_moments(path, label=label)
    gamma_bg, st = fit_single_gamma_from_moments(mean, var, shape_min=shape_min)
    meta = {
        "status": st,
        "n_r2": n,
        "n_taxa": n_taxa,
        "mean": mean,
        "var": var,
        "shape_min": shape_min,
        "shape_raw": (mean / max(var / mean, 1e-5)) if mean > 0 else float("nan"),
        "shape": gamma_bg.shape,
        "scale": gamma_bg.scale,
    }
    return gamma_bg, meta


def _mu_floor(mode: str, r2: np.ndarray, gamma_bg: SingleGamma, min_mu_quantile: float = 0.70) -> float:
    """Return μ lower bound for Stage2; 0.0 means unconstrained."""
    if mode == "unconstrained":
        return 0.0
    if mode == "soft_mu":
        # soft: μ > Gamma mean
        return float(gamma_bg.mean) + 1e-6
    if mode == "constrained":
        q = float(np.quantile(r2, min_mu_quantile))
        floor = max(q, 1.25 * float(gamma_bg.mean), 0.05)
        return float(min(floor, 0.85))
    raise ValueError(f"unknown stage2 mode: {mode}")


def fit_gamma_gaussian_mixture(
    r2: np.ndarray,
    gamma_bg: SingleGamma,
    *,
    mode: str = "unconstrained",
    seed: int = 42,
    max_iter: int = 200,
    tol: float = 1e-7,
    min_mu_quantile: float = 0.70,
) -> tuple[GammaGaussianMix, dict]:
    """Stage2: fix Gamma; EM for w, μ, σ on ALL intersection r².

    mode:
      unconstrained — no μ floor
      soft_mu       — μ > Gamma mean each iter
      constrained   — μ >= max(q0.7, 1.25*kθ, 0.05)
    """
    r2 = r2[np.isfinite(r2) & (r2 >= 0.0) & (r2 <= 1.0)]
    meta = {"status": "too_few", "n_r2": int(r2.size), "n_iter": 0, "stage2_mode": mode}
    if r2.size < 50:
        return GammaGaussianMix(gamma_bg, float("nan"), float("nan"), float("nan")), meta
    if not np.isfinite(gamma_bg.shape) or not np.isfinite(gamma_bg.scale):
        meta["status"] = "bad_gamma"
        return GammaGaussianMix(gamma_bg, float("nan"), float("nan"), float("nan")), meta

    rng = np.random.default_rng(seed)
    mu_floor = _mu_floor(mode, r2, gamma_bg, min_mu_quantile=min_mu_quantile)
    meta["mu_floor"] = mu_floor

    w_gamma = 0.85
    hi = r2[r2 >= max(mu_floor, float(np.percentile(r2, 75)))]
    if hi.size < 10:
        hi = r2[r2 >= mu_floor] if mu_floor > 0 else r2
    if hi.size < 10:
        hi = r2[rng.choice(r2.size, size=min(10_000, r2.size), replace=False)]
    mu = float(np.median(hi))
    if mu_floor > 0:
        mu = max(mu, mu_floor)
    sd = max(float(np.std(hi)), 1e-4)
    if mode == "constrained":
        sd = float(np.clip(sd, 0.02, 0.25))

    p_gamma_comp = gamma_bg.pointwise(r2)
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        p_g = w_gamma * p_gamma_comp
        p_n = (1.0 - w_gamma) * norm.pdf(r2, mu, sd)
        denom = p_g + p_n + 1e-300
        resp_g = p_g / denom
        resp_n = p_n / denom

        w_lo, w_hi = (0.05, 0.95) if mode == "constrained" else (1e-6, 1.0 - 1e-6)
        w_new = float(np.clip(resp_g.mean(), w_lo, w_hi))
        mass_n = float(resp_n.sum())
        if mass_n > 1e-12:
            mu_new = float((resp_n * r2).sum() / mass_n)
            var_n = float((resp_n * (r2 - mu_new) ** 2).sum() / mass_n)
            sd_new = max(float(np.sqrt(max(var_n, 1e-12))), 1e-4)
            if mode == "constrained":
                sd_new = float(np.clip(sd_new, 0.02, 0.25))
        else:
            mu_new, sd_new = mu, sd

        if mu_floor > 0:
            mu_new = max(mu_new, mu_floor)

        dw = abs(w_new - w_gamma)
        dmu = abs(mu_new - mu)
        dsd = abs(sd_new - sd)
        w_gamma, mu, sd = w_new, mu_new, sd_new
        if dw < tol and dmu < tol and dsd < tol:
            break

    if not np.isfinite(mu) or not np.isfinite(sd) or sd <= 0:
        meta["status"] = "bad_gaussian"
        return GammaGaussianMix(gamma_bg, w_gamma, float("nan"), float("nan")), meta

    mix = GammaGaussianMix(gamma_bg, w_gamma, mu, sd)
    if r2.size > 2_000_000:
        idx = rng.choice(r2.size, 2_000_000, replace=False)
        ll_x = r2[idx]
    else:
        ll_x = r2
    pdf = mix.pdf(ll_x)
    loglik = float(np.mean(np.log(pdf + 1e-300)))
    meta.update(
        {
            "status": "ok",
            "n_iter": n_iter,
            "w_gamma": w_gamma,
            "w_gaussian": 1.0 - w_gamma,
            "gauss_mean": mu,
            "gauss_std": sd,
            "mean_loglik_report": loglik,
            "n_loglik_report": int(ll_x.size),
        }
    )
    return mix, meta


# backward-compatible alias
def fit_gamma_gaussian_mixture_unconstrained(*args, **kwargs):
    kwargs.setdefault("mode", "unconstrained")
    return fit_gamma_gaussian_mixture(*args, **kwargs)


def estimate_wg_em(
    r2: np.ndarray,
    mix: GammaGaussianMix,
    *,
    min_pairs: int = 50,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> dict:
    """Stage3: EM for mixing weight only; Γ(k,θ) and N(μ,σ) fixed."""
    r2 = r2[np.isfinite(r2) & (r2 >= 0.0) & (r2 <= 1.0)]
    n_pairs = int(r2.size)
    out = {
        "status": "too_few",
        "n_pairs": n_pairs,
        "w_gamma": float("nan"),
        "w_gaussian": float("nan"),
        "frac_gaussian": float("nan"),
        "frac_gamma": float("nan"),
        "loglik": float("nan"),
        "n_iter": 0,
    }
    if n_pairs < min_pairs:
        return out
    if not (
        np.isfinite(mix.gamma.shape)
        and np.isfinite(mix.gamma.scale)
        and np.isfinite(mix.gauss_mean)
        and np.isfinite(mix.gauss_std)
        and mix.gauss_std > 0
    ):
        out["status"] = "bad_params"
        return out

    p_gamma = mix.gamma.pointwise(r2)
    p_gauss = norm.pdf(r2, mix.gauss_mean, mix.gauss_std)
    w_g = 0.2
    n_iter = 0
    for n_iter in range(1, max_iter + 1):
        num = w_g * p_gauss
        den = (1.0 - w_g) * p_gamma + num + 1e-300
        resp = num / den
        w_new = float(np.clip(resp.mean(), 1e-6, 1.0 - 1e-6))
        if abs(w_new - w_g) < tol:
            w_g = w_new
            break
        w_g = w_new

    wg = float(np.clip(w_g, W_CLIP[0], W_CLIP[1]))
    mix_pdf = (1.0 - w_g) * p_gamma + w_g * p_gauss
    out.update(
        {
            "status": "ok",
            "w_gamma": 1.0 - w_g,
            "w_gaussian": w_g,
            "frac_gaussian": wg,
            "frac_gamma": 1.0 - wg,
            "loglik": float(np.sum(np.log(mix_pdf + 1e-300))),
            "n_iter": n_iter,
        }
    )
    return out


def save_gamma_tsv(gamma: SingleGamma, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("shape\tscale\tmean\tnote\n")
        f.write(f"{gamma.shape}\t{gamma.scale}\t{gamma.mean}\t{note}\n")


def save_mixture_tsv(mix: GammaGaussianMix, path: Path, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(
            "gamma_shape\tgamma_scale\tgamma_mean\t"
            "w_gamma\tw_gaussian\tgauss_mean\tgauss_std\tnote\n"
        )
        f.write(
            f"{mix.gamma.shape}\t{mix.gamma.scale}\t{mix.gamma.mean}\t"
            f"{mix.w_gamma}\t{mix.w_gaussian}\t{mix.gauss_mean}\t{mix.gauss_std}\t{note}\n"
        )


def load_mixture_tsv(path: Path) -> GammaGaussianMix:
    import csv

    row = next(csv.DictReader(path.open(encoding="utf-8"), delimiter="\t"))
    g = SingleGamma(float(row["gamma_shape"]), float(row["gamma_scale"]))
    return GammaGaussianMix(g, float(row["w_gamma"]), float(row["gauss_mean"]), float(row["gauss_std"]))
