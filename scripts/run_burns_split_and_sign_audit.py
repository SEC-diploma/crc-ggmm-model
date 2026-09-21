#!/usr/bin/env python3
"""Burns88 tumour vs adjacent prevalence sensitivity, and STAR gene-pair sign audit.

Uses local matrices (not 16S identity at scoring time).
Does not refit GGMM on subsets and does not run GSEA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path("/mnt/c/storage_F/septemberwork/0901SCIpaper")
sys.path.insert(0, str(ROOT / "code"))
from coggmm.evaluate import precision_recall_at_k  # noqa: E402
from coggmm.scores import prevalence  # noqa: E402

BURNS = Path("/mnt/d/microbe/Maywork/Burns88/final_matrices/genus_level")
CORR = Path("/mnt/c/storage_F/SCI/corr_pearson_and_residual_pc10")
AUDIT = ROOT / "tables" / "Table_crc_38_audit.tsv"
OUT = ROOT / "tables"
FIG = ROOT / "figures" / "decision"
CRC = ["Purcell2017", "Korea", "Burns88", "Colon_CRC"]
TOOLS = {
    "star": "genus_x_sample__after_star.tsv",
    "kraken": "genus_x_sample__kraken_microbiome.tsv",
    "minimap2": "genus_x_sample__after_minimap.tsv",
    "bowtie2": "genus_x_sample__after_bowtie2.tsv",
}


def _metrics(y: np.ndarray, s: np.ndarray, taxid: np.ndarray) -> dict:
    y = np.asarray(y, int)
    s = np.asarray(s, float)
    m = np.isfinite(s)
    y, s, taxid = y[m], s[m], np.asarray(taxid)[m]
    out = {
        "n": int(y.size),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()) if y.size else np.nan,
        "auroc": np.nan,
        "auprc": np.nan,
    }
    if y.size >= 8 and len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, s))
        out["auprc"] = float(average_precision_score(y, s))
    p20, _, _, _ = precision_recall_at_k(y, s, 20, taxid=taxid)
    p50, _, _, _ = precision_recall_at_k(y, s, 50, taxid=taxid)
    out["precision_at_20"] = p20
    out["precision_at_50"] = p50
    return out


def burns_split() -> pd.DataFrame:
    s16 = pd.read_csv(BURNS / "genus_x_sample_16s_readcounts.tsv", sep="\t")
    s16_id = s16.columns[0]
    s16 = s16.set_index(s16_id)
    s16.index = pd.to_numeric(s16.index, errors="coerce")
    s16 = s16.loc[s16.index.notna()]
    s16.index = s16.index.astype(int)
    rows = []
    for tool, fn in TOOLS.items():
        M = pd.read_csv(BURNS / fn, sep="\t")
        M = M.set_index(M.columns[0])
        M.index = pd.to_numeric(M.index, errors="coerce")
        M = M.loc[M.index.notna()]
        M.index = M.index.astype(int)
        samples = [c for c in M.columns if c in s16.columns]
        tumor = [c for c in samples if c.endswith("_tumor")]
        adj = [c for c in samples if c.endswith("_normal")]
        for label, cols in (("all", samples), ("tumor", tumor), ("adjacent", adj)):
            Mc = M[cols]
            Sc = s16.reindex(columns=cols).fillna(0.0)
            prev = prevalence(Mc.to_numpy(float))
            rna_det = (Mc.to_numpy(float) > 0).any(axis=1)
            s16_on = (Sc.to_numpy(float) > 0).any(axis=1)
            s16_det = pd.Series(s16_on, index=Sc.index)
            tax = Mc.index.to_numpy()
            y = np.array([1 if (int(t) in s16_det.index and bool(s16_det.loc[int(t)])) else 0 for t in tax], int)
            keep = rna_det
            met = _metrics(y[keep], prev[keep], tax[keep])
            met.update(
                cohort="Burns88",
                tool=tool,
                subset=label,
                n_samples=len(cols),
                n_rna=int(keep.sum()),
                n_intersect=int(y[keep].sum()),
                intersect_frac=float(y[keep].mean()) if keep.any() else np.nan,
            )
            rows.append(met)
    return pd.DataFrame(rows)


def burns_38(audit: pd.DataFrame) -> pd.DataFrame:
    s16 = pd.read_csv(BURNS / "genus_x_sample_16s_readcounts.tsv", sep="\t")
    s16 = s16.set_index(s16.columns[0])
    s16.index = pd.to_numeric(s16.index, errors="coerce")
    s16 = s16.loc[s16.index.notna()]
    s16.index = s16.index.astype(int)
    star = pd.read_csv(BURNS / TOOLS["star"], sep="\t").set_index("taxid")
    star.index = pd.to_numeric(star.index, errors="coerce")
    star = star.loc[star.index.notna()]
    star.index = star.index.astype(int)
    samples = [c for c in star.columns if c in s16.columns]
    recs = []
    for _, g in audit.iterrows():
        tid = int(g.taxid)
        rec = {"taxid": tid, "genus": g.genus, "group": g.group}
        for label, cols in (
            ("all", samples),
            ("tumor", [c for c in samples if c.endswith("_tumor")]),
            ("adjacent", [c for c in samples if c.endswith("_normal")]),
        ):
            rna = bool(tid in star.index and (star.loc[tid, cols].to_numpy(float) > 0).any())
            splus = bool(tid in s16.index and (s16.loc[tid, cols].to_numpy(float) > 0).any())
            rec[f"{label}_rna"] = int(rna)
            rec[f"{label}_16s"] = int(splus)
            rec[f"{label}_dual"] = int(rna and splus)
            if tid in star.index:
                rec[f"{label}_prev"] = float((star.loc[tid, cols].to_numpy(float) > 0).mean())
            else:
                rec[f"{label}_prev"] = np.nan
        recs.append(rec)
    return pd.DataFrame(recs)


def sign_audit(audit: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-50 r² genes per 38-genus TaxID on STAR Pearson; cross-cohort sign agreement.

    Genes are selected from RNA r² only. 16S identity is not used.
    """
    tax38 = set(audit.taxid.astype(int))
    per_cell = []
    top_store = {}  # (cohort, taxid) -> list of (gene, r, r2, sign)
    for cohort in CRC:
        npz_path = CORR / cohort / "star" / "tables" / "pearson_r_r2_taxid_x_gene.npz"
        z = np.load(npz_path, allow_pickle=True)
        taxids = np.array(z["taxids"]).astype(int)
        genes = np.array(z["genes"]).astype(str)
        r = z["r"]
        r2 = z["r2"]
        for tid in sorted(tax38):
            hit = np.where(taxids == tid)[0]
            if hit.size == 0:
                continue
            i = int(hit[0])
            rr, rr2 = r[i], r2[i]
            m = np.isfinite(rr) & np.isfinite(rr2)
            order = np.argsort(-rr2[m])[:50]
            g = genes[m][order]
            rv = rr[m][order]
            r2v = rr2[m][order]
            signs = np.sign(rv)
            top_store[(cohort, tid)] = list(zip(g.tolist(), rv.tolist(), r2v.tolist(), signs.tolist()))
            per_cell.append(
                {
                    "cohort": cohort,
                    "taxid": tid,
                    "n_top": int(len(g)),
                    "n_pos_r": int((signs > 0).sum()),
                    "n_neg_r": int((signs < 0).sum()),
                    "frac_pos_r": float((signs > 0).mean()) if len(g) else np.nan,
                    "median_r": float(np.median(rv)) if len(g) else np.nan,
                    "median_r2": float(np.median(r2v)) if len(g) else np.nan,
                }
            )
        del z, r, r2
    per = pd.DataFrame(per_cell)

    # Cross-cohort: genes in top-50 of >=2 CRC STAR cells, same TaxID
    cross_rows = []
    for tid in sorted(tax38):
        gene_signs = {}
        for cohort in CRC:
            for gene, rv, r2v, sg in top_store.get((cohort, tid), []):
                gene_signs.setdefault(gene, []).append((cohort, sg, rv))
        shared = {g: v for g, v in gene_signs.items() if len(v) >= 2}
        n_shared = len(shared)
        n_agree = 0
        for g, v in shared.items():
            sgs = [x[1] for x in v]
            if len(set(int(np.sign(s)) if s != 0 else 0 for s in sgs) - {0}) <= 1:
                n_agree += 1
        genus = audit.loc[audit.taxid.astype(int) == tid, "genus"]
        cross_rows.append(
            {
                "taxid": tid,
                "genus": genus.iloc[0] if len(genus) else "",
                "n_cohorts_with_taxid": sum((c, tid) in top_store for c in CRC),
                "n_genes_top50_in_ge2_cohorts": n_shared,
                "n_sign_agree": n_agree,
                "frac_sign_agree": (n_agree / n_shared) if n_shared else np.nan,
            }
        )
    return per, pd.DataFrame(cross_rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    audit = pd.read_csv(AUDIT, sep="\t")
    split = burns_split()
    split.to_csv(OUT / "Table_burns88_tumor_adjacent.tsv", sep="\t", index=False)
    g38 = burns_38(audit)
    g38.to_csv(OUT / "Table_burns88_38_tumor_adjacent.tsv", sep="\t", index=False)
    per, cross = sign_audit(audit)
    per.to_csv(OUT / "Table_crc_38_star_top50_signs.tsv", sep="\t", index=False)
    cross.to_csv(OUT / "Table_crc_38_star_sign_consistency.tsv", sep="\t", index=False)

    # compact figure: prevalence AUPRC tumor vs adjacent vs all
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    tools = ["star", "minimap2", "bowtie2", "kraken"]
    x = np.arange(len(tools))
    w = 0.25
    colors = {"all": "#1B365D", "tumor": "#A34B2A", "adjacent": "#2F6B4F"}
    for i, sub in enumerate(("all", "tumor", "adjacent")):
        ys = [float(split[(split.tool == t) & (split.subset == sub)].auprc.iloc[0]) for t in tools]
        ax.bar(x + (i - 1) * w, ys, w, label=sub, color=colors[sub], edgecolor="white", linewidth=0.4)
    ax.set_xticks(x)
    ax.set_xticklabels(["STAR", "minimap2", "bowtie2", "Kraken-all"])
    ax.set_ylabel("AUPRC (prevalence)")
    ax.set_ylim(0, 0.55)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Burns88: prevalence ranking of 16S support after splitting tissue type")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png", "svg"):
        fig.savefig(FIG / f"Fig_burns88_tumor_adjacent.{ext}")
    plt.close(fig)
    print(split.to_string(index=False))
    print("38 dual tumor", int(g38.tumor_dual.sum()), "adjacent", int(g38.adjacent_dual.sum()), "all", int(g38.all_dual.sum()))
    print("sign agree median", float(cross.frac_sign_agree.median(skipna=True)))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
