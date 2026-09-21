#!/usr/bin/env python3
"""Stage 04 — align host, 16S and four RNA-microbe matrices on shared samples."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def align_cohort(host: Path, s16: Path, tools: dict[str, Path], out: Path) -> None:
    H = pd.read_csv(host, sep="\t", index_col=0)
    S = pd.read_csv(s16, sep="\t", index_col=0)
    samples = sorted(set(H.columns) & set(S.columns))
    if not samples:
        raise SystemExit("no shared samples between host and 16S")
    out.mkdir(parents=True, exist_ok=True)
    H[samples].to_csv(out / "gene_x_sample_expression.tsv", sep="\t")
    S[samples].to_csv(out / "taxid_x_sample_16s_readcounts.tsv", sep="\t")
    for name, path in tools.items():
        M = pd.read_csv(path, sep="\t", index_col=0)
        cols = [c for c in samples if c in M.columns]
        M[cols].to_csv(out / f"taxid_x_sample__{name}.tsv", sep="\t")
    print(f"wrote {out} n_samples={len(samples)}")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--host", required=True)
    p.add_argument("--s16", required=True)
    p.add_argument("--kraken", required=True)
    p.add_argument("--minimap2", required=True)
    p.add_argument("--bowtie2", required=True)
    p.add_argument("--star", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    align_cohort(
        Path(a.host),
        Path(a.s16),
        {
            "kraken_microbiome": a.kraken,
            "after_minimap": a.minimap2,
            "after_bowtie2": a.bowtie2,
            "after_star": a.star,
        },
        Path(a.out),
    )
