#!/usr/bin/env bash
set -euo pipefail

module load trf/4.09

PROJECT=/project/logsdon_shared/projects/carine/MUCIN1genotyping
LOCI_DIR="${PROJECT}/loci/oriented"
OUT_DIR="${PROJECT}/repeat_analysis/trf"
MANIFEST="${PROJECT}/metadata/assemblies.tsv"

mkdir -p "${OUT_DIR}"

SAMPLE=$(awk -v idx="${LSB_JOBINDEX}" 'NR==idx+1 {print $1}' "${MANIFEST}")
HAPLOTYPE=$(awk -v idx="${LSB_JOBINDEX}" 'NR==idx+1 {print $2}' "${MANIFEST}")
LABEL="${SAMPLE}_${HAPLOTYPE}"

FASTA="${LOCI_DIR}/${LABEL}.MUC1_10kb_padding.oriented.fa"

if [[ ! -f "${FASTA}" ]]; then
    echo "ERROR: FASTA not found: ${FASTA}"
    exit 1
fi

echo "Processing: ${LABEL}"

trf "${FASTA}" 2 7 7 80 10 50 500 -d -h -ngs \
    > "${OUT_DIR}/${LABEL}.trf.dat"

echo "Done: ${LABEL}"
