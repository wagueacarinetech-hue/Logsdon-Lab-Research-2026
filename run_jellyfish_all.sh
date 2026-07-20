#!/usr/bin/env bash
set -euo pipefail

PROJECT=/project/logsdon_shared/projects/carine/MUCIN1genotyping
LOCI_DIR="${PROJECT}/loci/oriented"
OUT_DIR="${PROJECT}/repeat_analysis/jellyfish"
KMER_FILE="${PROJECT}/repeat_analysis/jellyfish/VNTR_kmers_only.txt"
MANIFEST="${PROJECT}/metadata/assemblies.tsv"

mkdir -p "${OUT_DIR}"
mkdir -p "${PROJECT}/results"

SAMPLE=$(awk -v idx="${LSB_JOBINDEX}" 'NR==idx+1 {print $1}' "${MANIFEST}")
HAPLOTYPE=$(awk -v idx="${LSB_JOBINDEX}" 'NR==idx+1 {print $2}' "${MANIFEST}")
CONTINENTAL=$(awk -v idx="${LSB_JOBINDEX}" 'NR==idx+1 {print $3}' "${MANIFEST}")
POPULATION=$(awk -v idx="${LSB_JOBINDEX}" 'NR==idx+1 {print $4}' "${MANIFEST}")
LABEL="${SAMPLE}_${HAPLOTYPE}"

echo "Processing: ${LABEL}"

FASTA="${LOCI_DIR}/${LABEL}.MUC1_10kb_padding.oriented.fa"

if [[ ! -f "${FASTA}" ]]; then
    echo "ERROR: FASTA not found: ${FASTA}"
    exit 1
fi

jellyfish count \
    -m 21 \
    -s 100M \
    -t 8 \
    -C \
    "${FASTA}" \
    -o "${OUT_DIR}/${LABEL}.jf"

jellyfish query \
    "${OUT_DIR}/${LABEL}.jf" \
    $(cat "${KMER_FILE}" | tr '\n' ' ') \
    > "${OUT_DIR}/${LABEL}.VNTR_counts.txt"

COPY_NUMBER=$(awk '{sum += $2; count++} END {printf "%.2f", sum/count}' \
    "${OUT_DIR}/${LABEL}.VNTR_counts.txt")

echo -e "${SAMPLE}\t${HAPLOTYPE}\t${CONTINENTAL}\t${POPULATION}\t${COPY_NUMBER}" \
    >> "${PROJECT}/results/MUC1_copy_numbers.tsv"

echo "Done: ${LABEL} — Copy number: ${COPY_NUMBER}"
