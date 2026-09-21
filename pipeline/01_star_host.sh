#!/usr/bin/env bash
# Stage 01 — host gene expression (bulk RNA-seq)
# Usage: bash 01_star_host.sh SAMPLE_ID R1.fastq.gz R2.fastq.gz
set -euo pipefail
source "$(dirname "$0")/config.sh"
SAMPLE="${1:?sample id}"
R1="${2:?R1}"
R2="${3:?R2}"
OUT="${HOST_OUT}/${SAMPLE}"
mkdir -p "${OUT}"
STAR --runThreadN "${THREADS}" \
  --genomeDir "${STAR_GENOME}" \
  --readFilesIn "${R1}" "${R2}" \
  --readFilesCommand zcat \
  --outSAMtype BAM SortedByCoordinate \
  --quantMode GeneCounts \
  --outFileNamePrefix "${OUT}/"
# ReadsPerGene.out.tab column 2 = unstranded counts → gene × sample matrix later
