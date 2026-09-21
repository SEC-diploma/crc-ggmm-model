#!/usr/bin/env bash
# Stage 02d — STAR unmapped reads → Kraken2
set -euo pipefail
source "$(dirname "$0")/config.sh"
SAMPLE="${1:?}"
STAR_DIR="${HOST_OUT}/${SAMPLE}"
OUT="${MICRO_OUT}/star/${SAMPLE}"; mkdir -p "${OUT}"
# Convert STAR Unmapped.out.mate[12] to FASTQ if needed
kraken2 --db "${KRAKEN_DB}" --threads "${THREADS}" --paired \
  --report "${OUT}/kreport.txt" --output "${OUT}/kraken.out" \
  "${STAR_DIR}/Unmapped.out.mate1" "${STAR_DIR}/Unmapped.out.mate2"
python "$(dirname "$0")/kreport_to_taxid_counts.py" "${OUT}/kreport.txt" "${OUT}/taxid_counts.tsv"
