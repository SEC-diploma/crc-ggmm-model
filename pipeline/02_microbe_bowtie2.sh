#!/usr/bin/env bash
# Stage 02c — bowtie2 host depletion, then Kraken2 on unmapped reads
set -euo pipefail
source "$(dirname "$0")/config.sh"
SAMPLE="${1:?}"; R1="${2:?}"; R2="${3:?}"
OUT="${MICRO_OUT}/bowtie2/${SAMPLE}"; mkdir -p "${OUT}"
bowtie2 -x "${HOST_BT2}" -1 "${R1}" -2 "${R2}" -p "${THREADS}" \
  | samtools view -b -f 4 - \
  | samtools fastq -1 "${OUT}/unmap_R1.fq.gz" -2 "${OUT}/unmap_R2.fq.gz" -0 /dev/null -s /dev/null -
kraken2 --db "${KRAKEN_DB}" --threads "${THREADS}" --paired \
  --report "${OUT}/kreport.txt" --output "${OUT}/kraken.out" \
  "${OUT}/unmap_R1.fq.gz" "${OUT}/unmap_R2.fq.gz"
python "$(dirname "$0")/kreport_to_taxid_counts.py" "${OUT}/kreport.txt" "${OUT}/taxid_counts.tsv"
