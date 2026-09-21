# FASTQ → matrix pipeline (bulk + INVADEseq)

This folder documents the **exact stage order** used to build the eight bulk
cohorts and the INVADEseq OSCC single-cell atlas. Scripts are runnable templates;
reference genomes, Kraken2 databases and FASTQ paths must be set in `config.sh`.

## Bulk (eight dual-assay cohorts)

```
FASTQ (RNA-seq + 16S)
  ├─ 01_star_host.sh          STAR → host gene × sample counts (ENSG)
  ├─ 02_microbe_kraken.sh     Kraken2 on all RNA reads
  ├─ 02_microbe_minimap2.sh   minimap2 host depletion → Kraken2
  ├─ 02_microbe_bowtie2.sh    bowtie2 host depletion → Kraken2
  ├─ 02_microbe_star.sh       STAR unmapped → Kraken2
  ├─ 03_16s_dada2.R           DADA2 + SILVA → 16S taxid × sample
  └─ 04_build_matrices.py     align samples, write final_matrices/
```

Four microbial RNA tools are **independent cameras** on the same reads, not a
serial PRISM cascade. Kraken sees the largest catalogue (more false positives);
STAR / minimap2 / bowtie2 are tighter after host depletion.

Downstream CoGGMM scoring (this repository, `coggmm/`) starts from
`final_matrices/` and **does not refit 16S labels**.

## Single-cell (Galeano Niño *Nature* 2022, PRJNA811533)

```
10x GEX FASTQ  → STARsolo → gene × barcode
10x GEX FASTQ  → four microbe branches → taxid × barcode
16S FASTQ      → Cell Ranger barcode map + PathSeq/DADA2
Stage 05       → pseudobulk, correlation, GMM/Gamma deconvolution
```

Primary analysis: OSCC_11–15. Secondary (OSCC_16–17) has weak 16S and is QC only.

See `singlecell/invade_seq_scores.py` for the downstream adapter.
