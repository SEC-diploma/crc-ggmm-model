"""CoGGMM: complementary Gamma–Gaussian mixture scoring of dual-assay tissue microbes.

RNA-seq and 16S are both treated as noisy. Mixture parameters are fit from RNA
host–microbe r² only; 16S labels are reserved for evaluation and concordance
partitioning, never for EM.
"""

from .mixture import (
    W_CLIP,
    GammaGaussianMix,
    SingleGamma,
    estimate_wg_em,
    fit_gamma_gaussian_mixture,
    fit_single_gamma_from_moments,
    fit_single_gamma_full,
    load_mixture_tsv,
    save_gamma_tsv,
    save_mixture_tsv,
)
from .scores import (
    bayes_posterior,
    fraction_from_r2,
    lex_minus,
    pnxfraction_from_r2,
    prevalence,
    tooladapt_score,
)
from .evaluate import auprc, auroc, roc_points
from .fuse import concordance_classes, dual_supported_mask
from .preprocess import joint_logcpm

__all__ = [
    "W_CLIP",
    "GammaGaussianMix",
    "SingleGamma",
    "estimate_wg_em",
    "fit_gamma_gaussian_mixture",
    "fit_single_gamma_from_moments",
    "fit_single_gamma_full",
    "load_mixture_tsv",
    "save_gamma_tsv",
    "save_mixture_tsv",
    "bayes_posterior",
    "fraction_from_r2",
    "lex_minus",
    "pnxfraction_from_r2",
    "prevalence",
    "tooladapt_score",
    "auprc",
    "auroc",
    "roc_points",
    "concordance_classes",
    "dual_supported_mask",
    "joint_logcpm",
]
