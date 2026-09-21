#!/usr/bin/env python3
"""CRC 16-cell decision metrics for the hierarchical-evidence story.

Does not refit GGMM or change τ / correlation / cohort locks.
Reads locked fused scores (0825, host_all × Pearson × τ=0.6) and PC scores (0828).

Outputs under SCIchugao0826/tables/decision and figures/decision.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/mnt/c/storage_F/August-work/SCIchugao0826")
AUGUST = Path("/mnt/c/storage_F/August-work")
sys.path.insert(0, str(ROOT / "code"))
from coggmm.evaluate import auprc, auroc, precision_recall_at_k  # noqa: E402
from coggmm.scores import lex_minus  # noqa: E402

TAB = ROOT / "tables" / "decision"
FIG = ROOT / "figures" / "decision"
SFIG = ROOT / "supplementary" / "figures" / "decision"
for p in (TAB, FIG, SFIG):
    p.mkdir(parents=True, exist_ok=True)

CRC = ["Purcell2017", "Korea", "Burns88", "Colon_CRC"]
TOOLS = ["star", "kraken", "minimap2", "bowtie2"]
ALIGNERS = ["star", "minimap2", "bowtie2"]
SCORES = ["prevalence", "fraction", "pnxfraction", "tooladapt"]
KS = [10, 20, 50, 100]
N_BOOT = 500
N_PERM = 200
SEED = 20260831
KOREA_KRAKEN = ("Korea", "kraken")

PREV, FRAC, PNX, ADAPT = "#C62828", "#1565C0", "#2E7D32", "#6A1B9A"
COLORS = {
    "prevalence": PREV,
    "fraction": FRAC,
    "pnxfraction": PNX,
    "pnx_mean": PNX,
    "tooladapt": ADAPT,
    "tooladapt_shuffled": "#9E9E9E",
}
LABELS = {
    "prevalence": "Prevalence",
    "fraction": "GGMM fraction",
    "pnxfraction": "Pnx",
    "pnx_mean": "Pnx",
    "tooladapt": "tooladapt",
    "tooladapt_shuffled": "tooladapt (shuffled Pnx)",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)

AUDIT_NOTE = {
    848: "F. nucleatum / Fna C2: CRC enrichment and tumour invasion (Bullman 2017; Zepeda-Rivera 2024).",
    836: "Oral anaerobe repeatedly reported in CRC tissue and faeces.",
    838: "Oral genus; mixed CRC evidence, not equivalent to F. nucleatum.",
    32067: "Oral anaerobe co-occurring with Fusobacterium in some CRC studies.",
    543311: "Oral–gut taxon enriched in CRC (Parvimonas micra).",
    1257: "Oral anaerobe; P. anaerobius reported in CRC.",
    1378: "Oral coccus; G. morbillorum linked to CRC in several cohorts.",
    1301: "Common oral/gut genus; high prevalence, low specificity.",
    1016: "Oral genus; occasional CRC co-occurrence, weaker singleton evidence.",
    194: "Includes C. concisus / C. showae oral lineages; also enteric species.",
    157: "Oral spirochete; unusual in gut, treat as oral translocation candidate.",
    482: "Oral commensal; possible translocation, also kit-sensitive.",
    724: "Oral/respiratory; H. parainfluenzae sometimes in CRC datasets.",
    1654: "Oral Actinomyces; translocation vs dental carry-over unresolved.",
    86331: "Oral anaerobe in periodontal communities.",
    39948: "Oral/gut associated; D. pneumosintes in periodontal samples.",
    816: "Dominant gut genus; expected in tissue RNA if faecal carry-over or true mucosa.",
    239759: "Gut genus; bile-tolerant, often faecal rather than oral.",
    375288: "Gut genus frequently co-detected with Bacteroides.",
    216851: "Butyrate producer; dual-support may reflect residual lumen RNA/DNA.",
    572511: "Common Lachnospiraceae gut genus.",
    189330: "Gut Lachnospiraceae; not a CRC-specific marker.",
    207244: "Gut butyrate producer.",
    1263: "Gut genus (taxonomy unstable vs Mediterraneibacter).",
    1766253: "Gut Lachnospiraceae; 16S/RNA name mapping can be brittle.",
    1506553: "Gut Clostridium-cluster XIVa name; database-dependent.",
    1649459: "Gut genus; H. hathewayi reported in some CRC faecal studies.",
    946234: "Gut genus; often low abundance.",
    102106: "Actinobacterial gut genus; diet-associated.",
    1678: "Gut/probiotic genus; dual-support does not imply tumour tropism.",
    1350: "Gut/opportunistic; also hospital environment.",
    970: "Oral/gut associated; S. sputigena in periodontal samples.",
    165779: "Skin/oral anaerobe; possible body-site contamination of tissue.",
    570: "Gut pathobiont and environmental isolate; do not treat as kit-only.",
    286: "Environment, water systems, and reagent blanks (Salter/Eisenhofer); also opportunistic.",
    642: "Water-associated; treat as contamination-priority until orthogonal assay.",
    40323: "Water/reagent taxon frequently flagged in low-biomass studies.",
    13687: "Classic kit/reagent contaminant in 16S and RNA-seq (Salter 2014; Eisenhofer 2025).",
}


def _iqr(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    q75, q25 = np.percentile(x, [75, 25])
    return float(q75 - q25)


def _ci(x: np.ndarray, alpha: float = 0.05) -> tuple[float, float]:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(x, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def load_fused() -> pd.DataFrame:
    df = pd.read_csv(AUGUST / "0825work" / "tables" / "scores_fused_all.tsv", sep="\t")
    return df[df.cohort.isin(CRC)].copy()


def load_38() -> pd.DataFrame:
    return pd.read_csv(ROOT / "tables" / "Table_crc_dual_supported_38.tsv", sep="\t")


def metrics_one(y, s, taxid, ks=KS) -> dict:
    row = {
        "n": int(len(y)),
        "n_pos": int(np.asarray(y).sum()),
        "n_neg": int(len(y) - np.asarray(y).sum()),
        "base_rate": float(np.mean(y)),
        "auroc": auroc(y, s),
        "auprc": auprc(y, s),
    }
    for k in ks:
        p, r, n_hit, k_eff = precision_recall_at_k(y, s, k, taxid=taxid)
        row[f"precision_at_{k}"] = p
        row[f"recall_at_{k}"] = r
        row[f"n_pos_at_{k}"] = n_hit
        row[f"k_eff_{k}"] = k_eff
    return row


def cell_metrics(df: pd.DataFrame, score_cols=SCORES) -> pd.DataFrame:
    rows = []
    for (coh, tool), g in df.groupby(["cohort", "tool"], sort=False):
        y = g.y.to_numpy(int)
        taxid = g.genus_taxid.to_numpy()
        for sc in score_cols:
            rec = {"cohort": coh, "tool": tool, "score": sc}
            rec.update(metrics_one(y, g[sc].to_numpy(float), taxid))
            rows.append(rec)
    out = pd.DataFrame(rows)
    out["cohort"] = pd.Categorical(out.cohort, CRC, ordered=True)
    out["tool"] = pd.Categorical(out.tool, TOOLS, ordered=True)
    out["score"] = pd.Categorical(out.score, score_cols, ordered=True)
    return out.sort_values(["cohort", "tool", "score"]).reset_index(drop=True)


def summarize_score(cells: pd.DataFrame, metric: str) -> pd.DataFrame:
    rows = []
    for sc, g in cells.groupby("score", observed=True):
        vals = g[metric].to_numpy(float)
        by_coh = g.groupby("cohort", observed=True)[metric].mean()
        rows.append(
            {
                "score": sc,
                "metric": metric,
                "n_cells": int(len(g)),
                "mean": float(np.mean(vals)),
                "median": float(np.median(vals)),
                "iqr": _iqr(vals),
                "min_cell": float(np.min(vals)),
                "max_cell": float(np.max(vals)),
                "worst_cohort": str(by_coh.idxmin()),
                "worst_cohort_mean": float(by_coh.min()),
                "best_cohort": str(by_coh.idxmax()),
                "best_cohort_mean": float(by_coh.max()),
            }
        )
    return pd.DataFrame(rows)


def winrate_table(cells: pd.DataFrame) -> pd.DataFrame:
    pairs = [
        ("prevalence", "fraction"),
        ("prevalence", "pnxfraction"),
        ("tooladapt", "prevalence"),
    ]
    metrics = ["auprc", "auroc", "precision_at_20", "precision_at_50"]
    wide = cells.pivot_table(
        index=["cohort", "tool"], columns="score", values=metrics, observed=True
    )
    rows = []
    for m in metrics:
        for a, b in pairs:
            da = wide[(m, a)] - wide[(m, b)]
            da = da.dropna()
            n = int(len(da))
            n_win = int((da > 0).sum())
            n_tie = int((da == 0).sum())
            n_loss = int((da < 0).sum())
            rows.append(
                {
                    "metric": m,
                    "winner": a,
                    "loser": b,
                    "n_cells": n,
                    "n_win": n_win,
                    "n_tie": n_tie,
                    "n_loss": n_loss,
                    "winrate": n_win / n if n else float("nan"),
                    "mean_delta": float(da.mean()),
                    "median_delta": float(da.median()),
                    "iqr_delta": _iqr(da.to_numpy()),
                    "min_delta": float(da.min()),
                    "max_delta": float(da.max()),
                }
            )
    return pd.DataFrame(rows)


def delta_by_cell(cells: pd.DataFrame) -> pd.DataFrame:
    wide = cells.pivot_table(
        index=["cohort", "tool"],
        columns="score",
        values=["auprc", "auroc", "precision_at_20", "precision_at_50"],
        observed=True,
    )
    rows = []
    for (coh, tool), _ in wide.iterrows():
        rec = {"cohort": coh, "tool": tool}
        for m in ["auprc", "auroc", "precision_at_20", "precision_at_50"]:
            rec[f"delta_{m}_tooladapt_minus_prev"] = float(
                wide.loc[(coh, tool), (m, "tooladapt")] - wide.loc[(coh, tool), (m, "prevalence")]
            )
            rec[f"delta_{m}_prev_minus_fraction"] = float(
                wide.loc[(coh, tool), (m, "prevalence")] - wide.loc[(coh, tool), (m, "fraction")]
            )
            rec[f"delta_{m}_prev_minus_pnx"] = float(
                wide.loc[(coh, tool), (m, "prevalence")] - wide.loc[(coh, tool), (m, "pnxfraction")]
            )
        rec["is_kraken"] = tool == "kraken"
        rows.append(rec)
    return pd.DataFrame(rows)


def stratified_idx(y: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y, int)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    ip = rng.choice(pos, size=pos.size, replace=True) if pos.size else np.array([], int)
    inn = rng.choice(neg, size=neg.size, replace=True) if neg.size else np.array([], int)
    return np.concatenate([ip, inn])


def bootstrap_cells(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified TaxID bootstrap. Returns per-cell CI and 16-cell-mean CI."""
    groups = []
    for (coh, tool), g in df.groupby(["cohort", "tool"], sort=False):
        groups.append((coh, tool, g.reset_index(drop=True)))

    mean_draws = {sc: {m: [] for m in ["auprc", "auroc", "precision_at_50"]} for sc in SCORES}
    delta_mean_auprc = []
    delta_mean_auroc = []
    korea_delta_auprc = []
    korea_delta_auroc = []
    cell_draws = {}

    for b in range(N_BOOT):
        cell_vals = {(sc, m): [] for sc in SCORES for m in ["auprc", "auroc", "precision_at_50"]}
        for coh, tool, g in groups:
            rng = np.random.default_rng(SEED + b + 10007 * (CRC.index(coh) + 1) + 17 * (TOOLS.index(tool) + 1))
            y = g.y.to_numpy(int)
            taxid = g.genus_taxid.to_numpy()
            idx = stratified_idx(y, rng)
            yb, tb = y[idx], taxid[idx]
            recs = {}
            for sc in SCORES:
                sb = g[sc].to_numpy(float)[idx]
                recs[sc] = {
                    "auprc": auprc(yb, sb),
                    "auroc": auroc(yb, sb),
                    "precision_at_50": precision_recall_at_k(yb, sb, 50, taxid=tb)[0],
                }
                for m, v in recs[sc].items():
                    cell_vals[(sc, m)].append(v)
                    cell_draws.setdefault((coh, tool, sc, m), []).append(v)
            d_auprc = recs["tooladapt"]["auprc"] - recs["prevalence"]["auprc"]
            d_auroc = recs["tooladapt"]["auroc"] - recs["prevalence"]["auroc"]
            if (coh, tool) == KOREA_KRAKEN:
                korea_delta_auprc.append(d_auprc)
                korea_delta_auroc.append(d_auroc)
        for sc in SCORES:
            for m in ["auprc", "auroc", "precision_at_50"]:
                mean_draws[sc][m].append(float(np.nanmean(cell_vals[(sc, m)])))
        delta_mean_auprc.append(
            float(np.nanmean(cell_vals[("tooladapt", "auprc")]) - np.nanmean(cell_vals[("prevalence", "auprc")]))
        )
        delta_mean_auroc.append(
            float(np.nanmean(cell_vals[("tooladapt", "auroc")]) - np.nanmean(cell_vals[("prevalence", "auroc")]))
        )
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b+1}/{N_BOOT}", flush=True)

    cell_rows = []
    for (coh, tool, sc, m), draws in cell_draws.items():
        arr = np.asarray(draws, float)
        lo, hi = _ci(arr)
        cell_rows.append(
            {
                "cohort": coh,
                "tool": tool,
                "score": sc,
                "metric": m,
                "boot_mean": float(np.mean(arr)),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_boot": int(arr.size),
            }
        )
    cell_ci = pd.DataFrame(cell_rows)

    mean_rows = []
    for sc in SCORES:
        for m, draws in mean_draws[sc].items():
            arr = np.asarray(draws, float)
            lo, hi = _ci(arr)
            mean_rows.append(
                {
                    "level": "mean16",
                    "cohort": "ALL",
                    "tool": "ALL",
                    "score": sc,
                    "metric": m,
                    "boot_mean": float(np.mean(arr)),
                    "ci95_lo": lo,
                    "ci95_hi": hi,
                    "n_boot": N_BOOT,
                }
            )
    for metric, draws in [
        ("delta_auprc_tooladapt_minus_prev", delta_mean_auprc),
        ("delta_auroc_tooladapt_minus_prev", delta_mean_auroc),
    ]:
        arr = np.asarray(draws, float)
        lo, hi = _ci(arr)
        mean_rows.append(
            {
                "level": "mean16",
                "cohort": "ALL",
                "tool": "ALL",
                "score": "tooladapt-prevalence",
                "metric": metric,
                "boot_mean": float(np.mean(arr)),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_boot": N_BOOT,
            }
        )
    for metric, draws in [
        ("delta_auprc_tooladapt_minus_prev", korea_delta_auprc),
        ("delta_auroc_tooladapt_minus_prev", korea_delta_auroc),
    ]:
        arr = np.asarray(draws, float)
        lo, hi = _ci(arr)
        mean_rows.append(
            {
                "level": "cell",
                "cohort": "Korea",
                "tool": "kraken",
                "score": "tooladapt-prevalence",
                "metric": metric,
                "boot_mean": float(np.mean(arr)),
                "ci95_lo": lo,
                "ci95_hi": hi,
                "n_boot": N_BOOT,
            }
        )
    return cell_ci, pd.DataFrame(mean_rows)


def primary_axis(g: pd.DataFrame) -> np.ndarray:
    if str(g.tool.iloc[0]) == "kraken":
        return g.prev_mean4.to_numpy(float)
    return g.prevalence.to_numpy(float)


def shuffle_ggmm(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    perm_rows = []
    for (coh, tool), g in df.groupby(["cohort", "tool"], sort=False):
        g = g.reset_index(drop=True)
        y = g.y.to_numpy(int)
        taxid = g.genus_taxid.to_numpy()
        prev = g.prevalence.to_numpy(float)
        pnx = g.pnxfraction.to_numpy(float)
        primary = primary_axis(g)
        true_adapt = g.tooladapt.to_numpy(float)
        rng = np.random.default_rng(SEED + 13 * (CRC.index(coh) + 1) + TOOLS.index(tool) + 1)
        shuf_auprc, shuf_auroc, shuf_p50 = [], [], []
        shuf_d_auprc, shuf_d_auroc = [], []
        prev_auprc = auprc(y, prev)
        prev_auroc = auroc(y, prev)
        true_auprc = auprc(y, true_adapt)
        true_auroc = auroc(y, true_adapt)
        true_p50 = precision_recall_at_k(y, true_adapt, 50, taxid=taxid)[0]
        prev_p50 = precision_recall_at_k(y, prev, 50, taxid=taxid)[0]
        for _ in range(N_PERM):
            q = rng.permutation(pnx)
            s = lex_minus(primary, q)
            ap = auprc(y, s)
            ar = auroc(y, s)
            p50 = precision_recall_at_k(y, s, 50, taxid=taxid)[0]
            shuf_auprc.append(ap)
            shuf_auroc.append(ar)
            shuf_p50.append(p50)
            shuf_d_auprc.append(ap - prev_auprc)
            shuf_d_auroc.append(ar - prev_auroc)
        rec = {
            "cohort": coh,
            "tool": tool,
            "n_perm": N_PERM,
            "auprc_prevalence": prev_auprc,
            "auprc_tooladapt": true_auprc,
            "auprc_shuffled_mean": float(np.mean(shuf_auprc)),
            "delta_auprc_true": true_auprc - prev_auprc,
            "delta_auprc_shuffled_mean": float(np.mean(shuf_d_auprc)),
            "delta_auprc_shuffled_ci_lo": _ci(np.asarray(shuf_d_auprc))[0],
            "delta_auprc_shuffled_ci_hi": _ci(np.asarray(shuf_d_auprc))[1],
            "auroc_prevalence": prev_auroc,
            "auroc_tooladapt": true_auroc,
            "auroc_shuffled_mean": float(np.mean(shuf_auroc)),
            "delta_auroc_true": true_auroc - prev_auroc,
            "delta_auroc_shuffled_mean": float(np.mean(shuf_d_auroc)),
            "precision50_prevalence": prev_p50,
            "precision50_tooladapt": true_p50,
            "precision50_shuffled_mean": float(np.mean(shuf_p50)),
            "true_beats_prev_auprc": int(true_auprc > prev_auprc),
            "shuf_mean_beats_prev_auprc": int(float(np.mean(shuf_auprc)) > prev_auprc),
            "true_delta_exceeds_shuf_ci": int(
                (true_auprc - prev_auprc) > _ci(np.asarray(shuf_d_auprc))[1]
            ),
        }
        rows.append(rec)
        perm_rows.append(
            pd.DataFrame(
                {
                    "cohort": coh,
                    "tool": tool,
                    "perm": np.arange(N_PERM),
                    "delta_auprc": shuf_d_auprc,
                    "delta_auroc": shuf_d_auroc,
                }
            )
        )
        print(f"  shuffle {coh} {tool}", flush=True)
    return pd.DataFrame(rows), pd.concat(perm_rows, ignore_index=True)


def pc_analysis() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    src = pd.read_csv(AUGUST / "0828huibao" / "tables" / "scores_all.tsv", sep="\t")
    src = src[src.cohort.isin(CRC)].copy()
    src["pnxfraction"] = src["pnx_mean"]
    # rebuild four-tool prevalence lookup (invariant across corr)
    pear = src[src["corr"] == "pearson"]
    prev_map: dict[tuple, float] = {}
    for row in pear.itertuples(index=False):
        prev_map[(row.cohort, row.tool, int(row.genus_taxid))] = float(row.prevalence)

    def prev_mean4_row(coh, taxid) -> float:
        vals = np.array([prev_map.get((coh, t, taxid), np.nan) for t in TOOLS], float)
        return float(np.nanmean(vals)) if np.isfinite(vals).any() else float("nan")

    frames = []
    for corr, gcorr in src.groupby("corr", sort=False):
        recs = []
        for (coh, tool), g in gcorr.groupby(["cohort", "tool"], sort=False):
            g = g.copy()
            taxids = g.genus_taxid.astype(int).to_numpy()
            own = g.prevalence.to_numpy(float)
            pnx = g.pnx_mean.to_numpy(float)
            if tool == "kraken":
                primary = np.array([prev_mean4_row(coh, int(t)) for t in taxids], float)
            else:
                primary = own
            g["tooladapt"] = lex_minus(primary, pnx)
            g["corr"] = corr
            recs.append(g)
        pc = pd.concat(recs, ignore_index=True)
        m = cell_metrics(pc, score_cols=["prevalence", "fraction", "pnxfraction", "tooladapt"])
        m["corr"] = corr
        frames.append(m)
    metrics = pd.concat(frames, ignore_index=True)

    # rank stability: Spearman and Jaccard of top-K, within cohort×tool
    stab_rows = []
    corrs = ["pearson", "pc_partial_5", "pc_partial_10"]
    score_map = {
        "prevalence": "prevalence",
        "fraction": "fraction",
        "pnx_mean": "pnx_mean",
    }
    wide_scores = {}
    for corr, gcorr in src.groupby("corr"):
        wide_scores[corr] = {
            (r.cohort, r.tool, int(r.genus_taxid)): r
            for r in gcorr.itertuples(index=False)
        }
    for coh in CRC:
        for tool in TOOLS:
            # taxids present in all three corrs (same catalogue)
            keys = [k[2] for k in wide_scores["pearson"] if k[0] == coh and k[1] == tool]
            for sc, attr in score_map.items():
                vecs = {}
                ys = None
                taxids = np.array(keys, int)
                for corr in corrs:
                    vecs[corr] = np.array(
                        [getattr(wide_scores[corr][(coh, tool, t)], attr) for t in keys],
                        float,
                    )
                ys = np.array(
                    [wide_scores["pearson"][(coh, tool, t)].y for t in keys], int
                )
                for a, b in [("pearson", "pc_partial_5"), ("pearson", "pc_partial_10"), ("pc_partial_5", "pc_partial_10")]:
                    m = np.isfinite(vecs[a]) & np.isfinite(vecs[b])
                    if m.sum() < 8:
                        rho = float("nan")
                    else:
                        rho = float(pd.Series(vecs[a][m]).corr(pd.Series(vecs[b][m]), method="spearman"))
                    rec = {
                        "cohort": coh,
                        "tool": tool,
                        "score": sc,
                        "corr_a": a,
                        "corr_b": b,
                        "spearman_rho": rho,
                    }
                    for k in (20, 50):
                        sa = set(taxids[np.lexsort((taxids, -vecs[a]))][:k])
                        sb = set(taxids[np.lexsort((taxids, -vecs[b]))][:k])
                        rec[f"jaccard_top{k}"] = len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan")
                    # dual-supported at q75 of this score
                    def dual(v):
                        thr = np.nanquantile(v, 0.75)
                        return set(taxids[(ys == 1) & (v >= thr)])

                    da, db = dual(vecs[a]), dual(vecs[b])
                    rec["jaccard_dual_q75"] = (
                        len(da & db) / len(da | db) if (da | db) else float("nan")
                    )
                    stab_rows.append(rec)
    stab = pd.DataFrame(stab_rows)

    mean_pc = (
        metrics.groupby(["corr", "score"], observed=True)[["auroc", "auprc", "precision_at_50"]]
        .mean()
        .reset_index()
    )
    return metrics, stab, mean_pc


def cross_branch(df: pd.DataFrame, dual38: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    present = (
        df.groupby(["cohort", "genus_taxid"])["tool"]
        .agg(lambda s: set(s.astype(str)))
        .reset_index(name="tools")
    )
    present["in_star"] = present.tools.map(lambda s: "star" in s).astype(int)
    present["in_minimap2"] = present.tools.map(lambda s: "minimap2" in s).astype(int)
    present["in_bowtie2"] = present.tools.map(lambda s: "bowtie2" in s).astype(int)
    present["in_kraken"] = present.tools.map(lambda s: "kraken" in s).astype(int)
    present["n_aligner"] = present[["in_star", "in_minimap2", "in_bowtie2"]].sum(axis=1)
    present["kraken_only"] = ((present.in_kraken == 1) & (present.n_aligner == 0)).astype(int)
    present["aligner_any"] = (present.n_aligner > 0).astype(int)

    counts = (
        present.groupby("cohort")
        .agg(
            n_taxid=("genus_taxid", "nunique"),
            n_kraken=("in_kraken", "sum"),
            n_star=("in_star", "sum"),
            n_minimap2=("in_minimap2", "sum"),
            n_bowtie2=("in_bowtie2", "sum"),
            n_kraken_only=("kraken_only", "sum"),
            n_all_three_aligners=("n_aligner", lambda s: int((s == 3).sum())),
        )
        .reindex(CRC)
        .reset_index()
    )
    counts["frac_kraken_only"] = counts.n_kraken_only / counts.n_taxid

    ids38 = set(dual38.taxid.astype(int))
    d38 = present[present.genus_taxid.isin(ids38)].copy()
    d38 = d38.merge(dual38.rename(columns={"taxid": "genus_taxid"})[["genus_taxid", "genus", "group"]], on="genus_taxid", how="left")
    summary38 = (
        d38.groupby(["genus_taxid", "genus", "group"], dropna=False)
        .agg(
            n_cohorts=("cohort", "nunique"),
            n_cohorts_star=("in_star", "sum"),
            n_cohorts_minimap2=("in_minimap2", "sum"),
            n_cohorts_bowtie2=("in_bowtie2", "sum"),
            n_cohorts_kraken=("in_kraken", "sum"),
            n_cohorts_kraken_only=("kraken_only", "sum"),
            min_n_aligner=("n_aligner", "min"),
        )
        .reset_index()
    )
    return counts, d38, summary38


def audit_table(dual38: pd.DataFrame, summary38: pd.DataFrame) -> pd.DataFrame:
    out = dual38.merge(
        summary38,
        left_on="taxid",
        right_on="genus_taxid",
        how="left",
        suffixes=("", "_x"),
    )
    out["literature_note"] = out.taxid.map(lambda t: AUDIT_NOTE.get(int(t), ""))
    out["orthogonal_priority"] = out.group.map(
        {
            "oral-CRC": "high — prior CRC/oral evidence; FISH/qPCR candidate",
            "gut": "medium — expected gut signal; colonisation vs lumen carry-over unresolved",
            "environmental": "audit-required — reagent/water/kit risk; do not treat as tissue census",
        }
    )
    out["exclude_as_kraken_only"] = (out.n_cohorts_kraken_only.fillna(0) > 0).astype(int)
    cols = [
        "taxid",
        "genus",
        "group",
        "orthogonal_priority",
        "literature_note",
        "n_cohorts",
        "n_cohorts_star",
        "n_cohorts_minimap2",
        "n_cohorts_bowtie2",
        "n_cohorts_kraken",
        "n_cohorts_kraken_only",
        "min_n_aligner",
        "exclude_as_kraken_only",
        "Purcell2017_prev",
        "Korea_prev",
        "Burns88_prev",
        "Colon_CRC_prev",
    ]
    return out[cols]


def dashboard(cells, summary, wins, deltas, boot_mean, shuf, pc_mean, branch_counts, audit) -> pd.DataFrame:
    def m16(score, metric):
        return float(cells.loc[cells.score == score, metric].mean())

    def wr(metric, winner, loser):
        hit = wins[(wins.metric == metric) & (wins.winner == winner) & (wins.loser == loser)].iloc[0]
        return hit

    def boot(score, metric):
        hit = boot_mean[(boot_mean.score == score) & (boot_mean.metric == metric) & (boot_mean.level == "mean16")]
        if hit.empty:
            return float("nan"), float("nan"), float("nan")
        r = hit.iloc[0]
        return float(r.boot_mean), float(r.ci95_lo), float(r.ci95_hi)

    wr_ap = wr("auprc", "prevalence", "fraction")
    wr_ta = wr("auprc", "tooladapt", "prevalence")
    wr_au = wr("auroc", "tooladapt", "prevalence")
    ap_p, ap_lo, ap_hi = boot("prevalence", "auprc")
    drow = boot_mean[
        (boot_mean.score == "tooladapt-prevalence")
        & (boot_mean.metric == "delta_auprc_tooladapt_minus_prev")
        & (boot_mean.level == "mean16")
    ].iloc[0]
    krow = boot_mean[
        (boot_mean.cohort == "Korea")
        & (boot_mean.tool == "kraken")
        & (boot_mean.metric == "delta_auprc_tooladapt_minus_prev")
    ].iloc[0]
    shuf_true = float(shuf.delta_auprc_true.mean())
    shuf_null = float(shuf.delta_auprc_shuffled_mean.mean())
    n38_kraken_only = int(audit.n_cohorts_kraken_only.fillna(0).sum())
    n38_aligner = int((audit.min_n_aligner == 3).sum())

    rows = [
        {
            "claim": "1_assays_not_replicates",
            "primary_metric": "RNA∩16S / RNA catalogue; Kraken vs STAR size",
            "result": "Locked 4.8–28.6% intersection; Kraken-all larger than STAR/minimap2/bowtie2",
            "status": "locked",
        },
        {
            "claim": "2_prevalence_primary",
            "primary_metric": "16-cell mean AUPRC + winrate vs fraction/Pnx + worst-cohort AUPRC",
            "result": (
                f"AUPRC prevalence {m16('prevalence','auprc'):.3f} "
                f"(boot 95% CI {ap_lo:.3f}–{ap_hi:.3f}) vs fraction {m16('fraction','auprc'):.3f} "
                f"vs Pnx {m16('pnxfraction','auprc'):.3f}; "
                f"winrate vs fraction {int(wr_ap.n_win)}/{int(wr_ap.n_cells)}; "
                f"worst cohort {summary[(summary.score=='prevalence')&(summary.metric=='auprc')].iloc[0].worst_cohort} "
                f"{summary[(summary.score=='prevalence')&(summary.metric=='auprc')].iloc[0].worst_cohort_mean:.3f}; "
                f"P@50 prevalence {m16('prevalence','precision_at_50'):.3f}"
            ),
            "status": "computed",
        },
        {
            "claim": "3_ggmm_second_axis",
            "primary_metric": "Fraction/Pnx AUPRC << prevalence; PC residualisation moves GGMM not prevalence",
            "result": (
                "See Table_crc_pc_mean.tsv: prevalence AUPRC invariant; fraction/Pnx move with PC5/PC10. "
                "PC2 scores were never fit (r2 cache has pc_partial_3, not PC2)."
            ),
            "status": "computed_pc0_5_10_no_pc2",
        },
        {
            "claim": "4_modest_kraken_refinement",
            "primary_metric": "ΔAUPRC(tooladapt−prevalence) + shuffle + branch split",
            "result": (
                f"True mean ΔAUPRC {shuf_true:.4f} vs shuffled-Pnx {shuf_null:.4f}; "
                f"AUPRC winrate {int(wr_ta.n_win)}/{int(wr_ta.n_cells)}; "
                f"AUROC winrate {int(wr_au.n_win)}/{int(wr_au.n_cells)}; "
                f"mean16 ΔAUPRC boot {drow.boot_mean:.4f} ({drow.ci95_lo:.4f}–{drow.ci95_hi:.4f}); "
                f"Korea Kraken ΔAUPRC boot {krow.boot_mean:.4f} ({krow.ci95_lo:.4f}–{krow.ci95_hi:.4f})"
            ),
            "status": "computed",
        },
        {
            "claim": "5_candidates_not_census",
            "primary_metric": "STAR ∩ RNA-high ∩ 16S+ ∩ 4 cohorts; drop Kraken-only; audit environmental genera",
            "result": (
                f"38 STAR dual-supported genera; {n38_aligner}/38 in all three host-depletion branches in every cohort; "
                f"kraken-only flags on 38-list: {n38_kraken_only}. "
                f"Environmental genera retained on the list and labelled audit-required."
            ),
            "status": "computed",
        },
    ]
    return pd.DataFrame(rows)


def _save(df: pd.DataFrame, name: str) -> None:
    p = TAB / name
    df.to_csv(p, sep="\t", index=False)
    print("wrote", p, df.shape)


def fig_auprc_bars(cells: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    x = np.arange(len(CRC))
    w = 0.18
    scores = SCORES
    for i, sc in enumerate(scores):
        means, sds = [], []
        for coh in CRC:
            v = cells[(cells.cohort == coh) & (cells.score == sc)].auprc.to_numpy()
            means.append(float(np.mean(v)))
            sds.append(float(np.std(v, ddof=1)))
        ax.bar(
            x + (i - 1.5) * w,
            means,
            w,
            yerr=sds,
            color=COLORS[sc],
            label=LABELS[sc],
            capsize=2,
            error_kw={"lw": 0.7},
        )
    ax.set_xticks(x, CRC)
    ax.set_ylabel("AUPRC (mean ± SD over 4 RNA branches)")
    ax.set_ylim(0, 0.65)
    ax.legend(frameon=False, ncol=2, fontsize=8)
    ax.set_title("Prevalence is the most stable RNA ranking of 16S support", loc="left", fontweight="bold")
    fig.tight_layout()
    for folder in (FIG, SFIG):
        fig.savefig(folder / "Fig_decision_auprc_by_cohort.png")
        fig.savefig(folder / "Fig_decision_auprc_by_cohort.pdf")
        fig.savefig(folder / "Fig_decision_auprc_by_cohort.svg")
    plt.close(fig)


def fig_precision_k(cells: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.8), sharey=True)
    for ax, k, title in zip(axes, [20, 50], ["Precision@20", "Precision@50"]):
        x = np.arange(len(CRC))
        w = 0.18
        col = f"precision_at_{k}"
        for i, sc in enumerate(SCORES):
            means, sds = [], []
            for coh in CRC:
                v = cells[(cells.cohort == coh) & (cells.score == sc)][col].to_numpy()
                means.append(float(np.mean(v)))
                sds.append(float(np.std(v, ddof=1)))
            ax.bar(
                x + (i - 1.5) * w,
                means,
                w,
                yerr=sds,
                color=COLORS[sc],
                label=LABELS[sc],
                capsize=2,
                error_kw={"lw": 0.7},
            )
        ax.set_xticks(x, CRC, rotation=15, ha="right")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylim(0, 1.0)
    axes[0].set_ylabel("Precision@K (mean ± SD over branches)")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for folder in (FIG, SFIG):
        fig.savefig(folder / "Fig_decision_precision_at_k.png")
        fig.savefig(folder / "Fig_decision_precision_at_k.pdf")
        fig.savefig(folder / "Fig_decision_precision_at_k.svg")
    plt.close(fig)


def fig_delta(deltas: pd.DataFrame, shuf: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 3.9))
    ax = axes[0]
    x = np.arange(len(CRC))
    w = 0.35
    for i, (mask, lab, c) in enumerate(
        [
            (deltas.is_kraken, "Kraken-all", "#6A1B9A"),
            (~deltas.is_kraken, "Host-depleted mean", "#546E7A"),
        ]
    ):
        means, sds = [], []
        for coh in CRC:
            sub = deltas[(deltas.cohort == coh) & mask]
            if lab.startswith("Host"):
                v = deltas[(deltas.cohort == coh) & (~deltas.is_kraken)].delta_auprc_tooladapt_minus_prev
            else:
                v = sub.delta_auprc_tooladapt_minus_prev
            means.append(float(v.mean()))
            sds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
        ax.bar(x + (i - 0.5) * w, means, w, yerr=sds, color=c, label=lab, capsize=2, error_kw={"lw": 0.7})
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(x, CRC, rotation=15, ha="right")
    ax.set_ylabel("ΔAUPRC (tooladapt − prevalence)")
    ax.set_title("Gain is a Kraken-catalogue effect", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    ax.scatter(shuf.delta_auprc_shuffled_mean, shuf.delta_auprc_true, c=["#6A1B9A" if t == "kraken" else "#546E7A" for t in shuf.tool], s=28, zorder=3)
    lims = [
        min(shuf.delta_auprc_shuffled_mean.min(), shuf.delta_auprc_true.min()) - 0.01,
        max(shuf.delta_auprc_shuffled_mean.max(), shuf.delta_auprc_true.max()) + 0.01,
    ]
    ax.plot(lims, lims, color="0.6", ls="--", lw=0.8)
    ax.axhline(0, color="0.8", lw=0.6)
    ax.axvline(0, color="0.8", lw=0.6)
    ax.set_xlabel("ΔAUPRC, shuffled Pnx (mean of 200 perms)")
    ax.set_ylabel("ΔAUPRC, true Pnx")
    ax.set_title("True GGMM exceeds a shuffled second axis", loc="left", fontweight="bold")
    fig.tight_layout()
    for folder in (FIG, SFIG):
        fig.savefig(folder / "Fig_decision_delta_auprc_shuffle.png")
        fig.savefig(folder / "Fig_decision_delta_auprc_shuffle.pdf")
        fig.savefig(folder / "Fig_decision_delta_auprc_shuffle.svg")
    plt.close(fig)


def fig_pc(pc_mean: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.8))
    order = ["pearson", "pc_partial_5", "pc_partial_10"]
    lab = {"pearson": "PC0 (Pearson)", "pc_partial_5": "PC5", "pc_partial_10": "PC10"}
    x = np.arange(len(order))
    w = 0.22
    for i, (sc, c) in enumerate([("prevalence", PREV), ("fraction", FRAC), ("pnxfraction", PNX)]):
        ys = [float(pc_mean[(pc_mean["corr"] == cr) & (pc_mean.score == sc)].auprc.iloc[0]) for cr in order]
        ax.bar(x + (i - 1) * w, ys, w, color=c, label=LABELS[sc])
    ax.set_xticks(x, [lab[c] for c in order])
    ax.set_ylabel("16-cell mean AUPRC")
    ax.set_ylim(0, 0.5)
    ax.set_title("Prevalence is PC-invariant; GGMM scores move", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    for folder in (FIG, SFIG):
        fig.savefig(folder / "Fig_decision_pc_auprc.png")
        fig.savefig(folder / "Fig_decision_pc_auprc.pdf")
        fig.savefig(folder / "Fig_decision_pc_auprc.svg")
    plt.close(fig)


def fig_branch(counts: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    x = np.arange(len(CRC))
    w = 0.22
    ax.bar(x - 1.5 * w, counts.n_star, w, color="#1B365D", label="STAR")
    ax.bar(x - 0.5 * w, counts.n_minimap2, w, color="#0B6B6B", label="minimap2")
    ax.bar(x + 0.5 * w, counts.n_bowtie2, w, color="#546E7A", label="bowtie2")
    ax.bar(x + 1.5 * w, counts.n_kraken, w, color="#C44E28", label="Kraken-all")
    ax.set_xticks(x, CRC)
    ax.set_ylabel("Genus TaxIDs detected")
    ax.set_title("Kraken-all inflates the catalogue relative to host-depleted branches", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=2, fontsize=8)
    fig.tight_layout()
    for folder in (FIG, SFIG):
        fig.savefig(folder / "Fig_decision_catalog_by_branch.png")
        fig.savefig(folder / "Fig_decision_catalog_by_branch.pdf")
        fig.savefig(folder / "Fig_decision_catalog_by_branch.svg")
    plt.close(fig)


def write_readme(meta: dict) -> None:
    text = f"""# CRC decision-metric outputs

Generated by `code/scripts/run_crc_decision_metrics.py`.

## Locks (not changed)
- Cohorts: Purcell2017, Korea, Burns88, Colon_CRC
- RNA branches: STAR, Kraken-all, minimap2, bowtie2
- Scores: prevalence, GGMM fraction, Pnx, tooladapt from 0825 fused table (host_all × Pearson × τ=0.6)
- 16S identity is a post-hoc label only

## What was computed
- Precision@K / Recall@K at K=10,20,50,100
- AUPRC/AUROC summaries (mean, median, IQR, worst cohort)
- WinRate on 16 cells
- Stratified TaxID bootstrap (n={meta['n_boot']}, seed={meta['seed']})
- Shuffled-Pnx tooladapt null (n_perm={meta['n_perm']})
- PC0/PC5/PC10 rank stability from 0828 scores (AUPRC, Spearman, Jaccard)
- Cross-branch detection and 38-genus audit

## What was not computed
- PC2: no fitted scores. 0821 r² cache has `pc_partial_3`, not PC2; Stage-3 GGMM was never run for PC2/PC3.
- Gene-level host–microbe pair sign consistency (needs r² matrices, not score tables)
- Pathway / GSEA (optional Fig. 6; not a primary decision metric)
- FISH, qPCR, MaAsLin2, T2T, decontam blanks

## Primary decision metrics by claim
1. Catalogue overlap and Kraken vs STAR size
2. 16-cell AUPRC, winrate, worst-cohort AUPRC; Precision@20/50 operational
3. Fraction/Pnx AUPRC vs prevalence; PC movement of GGMM
4. ΔAUPRC tooladapt−prevalence vs shuffled Pnx; Kraken vs aligner split
5. STAR dual-supported 38 genera, not Kraken-only, with contamination audit
"""
    (TAB / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    print("loading fused scores", flush=True)
    df = load_fused()
    dual38 = load_38()
    print(f"CRC score rows={len(df)}  38 genera={len(dual38)}", flush=True)

    print("cell metrics + P@K", flush=True)
    cells = cell_metrics(df)
    _save(cells, "Table_crc_metrics_by_cell.tsv")
    pk_long = cells.melt(
        id_vars=["cohort", "tool", "score", "n", "n_pos"],
        value_vars=[c for c in cells.columns if c.startswith("precision_at_") or c.startswith("recall_at_")],
        var_name="metric",
        value_name="value",
    )
    _save(pk_long, "Table_crc_precision_recall_at_k.tsv")

    summaries = pd.concat(
        [summarize_score(cells, m) for m in ["auprc", "auroc", "precision_at_20", "precision_at_50"]],
        ignore_index=True,
    )
    _save(summaries, "Table_crc_metric_summary.tsv")

    wins = winrate_table(cells)
    _save(wins, "Table_crc_winrate.tsv")
    deltas = delta_by_cell(cells)
    _save(deltas, "Table_crc_delta_by_cell.tsv")

    print(f"bootstrap n={N_BOOT}", flush=True)
    cell_ci, boot_mean = bootstrap_cells(df)
    _save(cell_ci, "Table_crc_bootstrap_by_cell.tsv")
    _save(boot_mean, "Table_crc_bootstrap_mean16.tsv")

    print(f"shuffle GGMM n_perm={N_PERM}", flush=True)
    shuf, shuf_perms = shuffle_ggmm(df)
    _save(shuf, "Table_crc_shuffle_ggmm.tsv")
    _save(shuf_perms, "Table_crc_shuffle_ggmm_perms.tsv")

    print("PC0/5/10 robustness", flush=True)
    pc_metrics, pc_stab, pc_mean = pc_analysis()
    _save(pc_metrics, "Table_crc_pc_metrics_by_cell.tsv")
    _save(pc_stab, "Table_crc_pc_rank_stability.tsv")
    _save(pc_mean, "Table_crc_pc_mean.tsv")

    print("cross-branch + audit", flush=True)
    branch_counts, branch38, summary38 = cross_branch(df, dual38)
    _save(branch_counts, "Table_crc_catalog_by_branch.tsv")
    _save(branch38, "Table_crc_38_by_cohort_branch.tsv")
    _save(summary38, "Table_crc_38_cross_branch.tsv")
    audit = audit_table(dual38, summary38)
    _save(audit, "Table_crc_38_audit.tsv")

    dash = dashboard(cells, summaries, wins, deltas, boot_mean, shuf, pc_mean, branch_counts, audit)
    _save(dash, "Table_crc_decision_dashboard.tsv")

    # also copy key tables next to locked CRC tables
    for name in [
        "Table_crc_metrics_by_cell.tsv",
        "Table_crc_metric_summary.tsv",
        "Table_crc_winrate.tsv",
        "Table_crc_bootstrap_mean16.tsv",
        "Table_crc_shuffle_ggmm.tsv",
        "Table_crc_pc_mean.tsv",
        "Table_crc_38_audit.tsv",
        "Table_crc_decision_dashboard.tsv",
        "Table_crc_catalog_by_branch.tsv",
        "Table_crc_precision_recall_at_k.tsv",
        "Table_crc_delta_by_cell.tsv",
        "Table_crc_pc_rank_stability.tsv",
    ]:
        src = TAB / name
        dst = ROOT / "tables" / name
        dst.write_bytes(src.read_bytes())

    print("figures", flush=True)
    fig_auprc_bars(cells)
    fig_precision_k(cells)
    fig_delta(deltas, shuf)
    fig_pc(pc_mean)
    fig_branch(branch_counts)

    meta = {
        "n_boot": N_BOOT,
        "n_perm": N_PERM,
        "seed": SEED,
        "n_crc_rows": int(len(df)),
        "ks": KS,
        "pc2": "not computed; no fitted scores",
        "pc_configs_used": ["pearson", "pc_partial_5", "pc_partial_10"],
    }
    (TAB / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    write_readme(meta)
    print(dash.to_string(index=False))
    print("done", TAB)


if __name__ == "__main__":
    main()
