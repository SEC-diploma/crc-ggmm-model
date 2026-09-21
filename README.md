# CRC dual-assay RNA-seq / 16S scoring

Code for ranking microbial signals in colorectal cancer tissue from paired
host RNA-seq and 16S rRNA data, including Kraken2, STAR, minimap2, Bowtie2,
host–microbe correlation, and a Gamma–Gaussian mixture on \(r^2\).

This repository contains scripts only. FASTQ files, STAR/Bowtie2/minimap2
indexes and Kraken2 databases are not included.

## Layout

```
pipeline/     STAR host counts; Kraken2 / minimap2 / Bowtie2 / STAR microbe
coggmm/       joint logCPM correlation; Gamma–Gaussian mixture; ranking scores
scripts/      CRC decision metrics and a small mixture demo
```

## 1. FASTQ pipeline

Edit paths in `pipeline/config.sh`, then:

```bash
bash pipeline/01_star_host.sh SAMPLE R1.fq.gz R2.fq.gz
bash pipeline/02_microbe_kraken.sh SAMPLE R1.fq.gz R2.fq.gz
bash pipeline/02_microbe_minimap2.sh SAMPLE R1.fq.gz R2.fq.gz
bash pipeline/02_microbe_bowtie2.sh SAMPLE R1.fq.gz R2.fq.gz
bash pipeline/02_microbe_star.sh SAMPLE
Rscript pipeline/03_16s_dada2.R
python pipeline/04_build_matrices.py
```

Kraken2 classifies all RNA reads. The minimap2, Bowtie2 and STAR microbe
scripts deplete host reads first, then run Kraken2 on the unmapped fraction.
STAR in `01_star_host.sh` is also used for host gene counts.

## 2. Correlation and Gamma–Gaussian mixture

```bash
python -m pip install -r requirements.txt
python scripts/run_coggmm_demo.py
python scripts/run_crc_decision_metrics.py
```

Core functions:

- `coggmm.preprocess.joint_logcpm` — library-size normalisation
- `coggmm.mixture.fit_gamma_gaussian_mixture` — Gamma background + Gaussian bump
- `coggmm.scores` — prevalence, fraction, Pnx, tooladapt

## Data

Public dual-assay CRC FASTQ are in NCBI SRA (see the manuscript Table of cohorts).
Do not upload raw reads or Kraken databases to this repository.
