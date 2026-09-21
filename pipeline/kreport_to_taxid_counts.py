#!/usr/bin/env python3
"""Convert a Kraken2 kreport to a taxid count table (species-level rows)."""
from __future__ import annotations

import argparse
from pathlib import Path


def kreport_to_taxid(kreport: Path, out: Path, rank: str = "S") -> None:
    rows = ["taxid\tcount"]
    with kreport.open() as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 6:
                continue
            # standard kraken2 report: pct, clade, assigned, rank, taxid, name
            assigned = int(float(parts[2]))
            rk = parts[3].strip()
            taxid = parts[4].strip()
            if rk.startswith(rank) and assigned > 0:
                rows.append(f"{taxid}\t{assigned}")
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("kreport")
    p.add_argument("out")
    args = p.parse_args()
    kreport_to_taxid(Path(args.kreport), Path(args.out))
