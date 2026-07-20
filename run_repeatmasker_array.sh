#!/usr/bin/env bash
set -euo pipefail
module load RepeatMasker/4.1.5
module load hmmer/3.1b2
PROJECT=/project/logsdon_shared/projects/carine/MUCIN1genotyping
LOCI_DIR="${PROJECT}/loci/oriented"
OUT_DIR="${PROJECT}/repeat_analysis/repeatmasker"
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
RepeatMasker -species human -dir "${OUT_DIR}" -pa 8 -engine hmmer "${FASTA}"
echo "Done: ${LABEL}"
