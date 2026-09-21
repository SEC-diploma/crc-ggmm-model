#!/usr/bin/env bash
# Stage 02a — Kraken2 on all RNA reads (largest catalogue, more false positives)
set -euo pipefail
source "$(dirname "$0")/config.sh"
SAMPLE="${1:?sample id}"; R1="${2:?}"; R2="${3:?}"
OUT="${MICRO_OUT}/kraken/${SAMPLE}"; mkdir -p "${OUT}"
kraken2 --db "${KRAKEN_DB}" --threads "${THREADS}" --paired \
  --report "${OUT}/kreport.txt" --output "${OUT}/kraken.out" \
  "${R1}" "${R2}"
python "$(dirname "$0")/kreport_to_taxid_counts.py" "${OUT}/kreport.txt" "${OUT}/taxid_counts.tsv"
