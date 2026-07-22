import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from scipy import stats

# ─── Configuration ───────────────────────────────────────────────────────────

TRF_DIR = "/Users/wagueacarinefongang/Desktop/trf"
JELLYFISH_RESULTS = "/Users/wagueacarinefongang/Desktop/MUC1_copy_numbers.tsv"
OUTPUT_PLOT = "/Users/wagueacarinefongang/Desktop/MUC1_jellyfish_vs_trf_v4.png"

MUC1_PERIOD_MIN = 55
MUC1_PERIOD_MAX = 65

# ─── Parse TRF output files ───────────────────────────────────────────────────
# Sum all contiguous 60bp records per sample
# Remove ALL low-copy low-identity boundary records
# These are flanking elements not true VNTR copies
# They appear in all 139 samples and inflate TRF estimates

results = []
all_records = []

for fname in os.listdir(TRF_DIR):
    if not fname.endswith(".dat"):
        continue

    parts = fname.split("_hap")
    sample = parts[0]
    haplotype = "hap" + parts[1].split(".")[0]

    with open(os.path.join(TRF_DIR, fname)) as f:
        content = f.read()

    records = []
    for line in content.split("\n"):
        fields = line.strip().split()
        if len(fields) < 14:
            continue
        try:
            start = int(fields[0])
            end = int(fields[1])
            period = int(fields[2])
            copy_number = float(fields[3])
            score = int(fields[7])
            pct_matches = int(fields[5])

            if MUC1_PERIOD_MIN <= period <= MUC1_PERIOD_MAX:
                records.append({
                    "start": start,
                    "end": end,
                    "copy_number": copy_number,
                    "score": score,
                    "pct_matches": pct_matches
                })
                all_records.append({
                    "sample": sample,
                    "haplotype": haplotype,
                    "start": start,
                    "end": end,
                    "copy_number": copy_number,
                    "score": score,
                    "pct_matches": pct_matches
                })
        except:
            continue

    if records:
        records.sort(key=lambda x: x["start"])

        # remove ALL boundary records with low copy number and low identity
        main_records = [r for r in records
                        if not (r["copy_number"] < 4 and r["pct_matches"] < 95)]

        # fall back to all records if filtering removes everything
        if not main_records:
            main_records = records

        total_copies = sum(r["copy_number"] for r in main_records)
        total_score = sum(r["score"] for r in main_records)
        n_records = len(main_records)
    else:
        total_copies = None
        total_score = 0
        n_records = 0

    results.append({
        "sample": sample,
        "haplotype": haplotype,
        "trf_copy_number": total_copies,
        "trf_score": total_score,
        "trf_n_records": n_records
    })

df_trf = pd.DataFrame(results)
df_records = pd.DataFrame(all_records)

# ─── Check boundary records ───────────────────────────────────────────────────

low_records = df_records[(df_records['copy_number'] < 4) &
                          (df_records['pct_matches'] < 95)]
n_unique = low_records.groupby(['sample', 'haplotype']).ngroups
print(f"Unique samples with boundary record: {n_unique}")
print(f"Total boundary records: {len(low_records)}")
print("\nSamples with multiple boundary records:")
print(low_records.groupby(['sample', 'haplotype']).size().sort_values(ascending=False).head(10))

# ─── Merge with Jellyfish results ────────────────────────────────────────────

jellyfish = pd.read_csv(JELLYFISH_RESULTS, sep="\t")
df = df_trf.merge(jellyfish, on=["sample", "haplotype"])
df = df[df['continental_group'] != 'REF']

# ─── QC check ────────────────────────────────────────────────────────────────

print(f"\nTotal samples: {df.shape[0]}")
missing = df[df['trf_copy_number'].isna()]
print(f"Samples where TRF found no MUC1 VNTR: {len(missing)}")

# ─── Correlation and statistics ───────────────────────────────────────────────

df_valid = df.dropna(subset=['trf_copy_number'])
r, p = stats.pearsonr(df_valid['trf_copy_number'], df_valid['copy_number'])
print(f"\nCorrelation TRF vs Jellyfish: r={r:.3f}, p={p:.6f}")

df_valid['ratio'] = df_valid['trf_copy_number'] / df_valid['copy_number']
df_valid['diff'] = df_valid['trf_copy_number'] - df_valid['copy_number']

print("\nFirst 10 samples sorted by Jellyfish copy number:")
print(df_valid[['sample', 'haplotype', 'copy_number', 'trf_copy_number',
                'ratio', 'diff']].sort_values('copy_number').head(10).to_string())

print("\nDifference statistics:")
print(df_valid[['ratio', 'diff']].describe().round(2).to_string())

print("\nTop 5 samples with highest ratio:")
print(df_valid[['sample', 'haplotype', 'continental_group', 'copy_number',
                'trf_copy_number', 'ratio']].sort_values('ratio', ascending=False).head(5).to_string())

print("\nTop 5 samples with lowest ratio:")
print(df_valid[['sample', 'haplotype', 'continental_group', 'copy_number',
                'trf_copy_number', 'ratio']].sort_values('ratio').head(5).to_string())

# ─── Verify NA20509 ───────────────────────────────────────────────────────────

print("\nNA20509 — most extreme heterozygote in dataset:")
print(jellyfish[jellyfish['sample'] == 'NA20509'][['sample', 'haplotype',
      'continental_group', 'population', 'copy_number']].to_string())

# ─── Plot ─────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 9))

sns.scatterplot(data=df_valid, x="copy_number", y="trf_copy_number",
                hue="continental_group", palette="Set2",
                alpha=0.7, ax=ax)

ax.plot([0, 130], [0, 130], 'k--', alpha=0.3, label="perfect agreement (y=x)")

slope, intercept, r, p, se = stats.linregress(df_valid['copy_number'],
                                                df_valid['trf_copy_number'])
x_range = np.linspace(0, 130, 100)
ax.plot(x_range, slope * x_range + intercept, 'r-',
        alpha=0.7, label=f"fitted line (slope={slope:.2f}, r={r:.2f})")

# annotate samples with extreme ratios
outliers = df_valid[(df_valid['ratio'] > 1.5) | (df_valid['ratio'] < 1.2)]
for _, row in outliers.iterrows():
    ax.annotate(f"{row['sample']}_{row['haplotype']}",
                xy=(row['copy_number'], row['trf_copy_number']),
                xytext=(5, 5), textcoords='offset points', fontsize=8)

ax.set_xlabel("Jellyfish copy number estimate")
ax.set_ylabel("TRF copy number estimate")
ax.set_title(f"Jellyfish vs TRF copy number estimates\nr={r:.3f}, slope={slope:.2f}")
ax.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=150)
plt.show()
print(f"\nslope={slope:.3f}, intercept={intercept:.3f}, r={r:.3f}")
print(f"Plot saved to {OUTPUT_PLOT}")

# find which sample is in boundary records but not in main dataframe
boundary_keys = set(df_records['sample'] + '_' + df_records['haplotype'])
main_keys = set(df['sample'] + '_' + df['haplotype'])
mystery = boundary_keys - main_keys
print("Sample in boundary records but not in main dataframe:")
print(mystery)

# show full list of samples with multiple boundary records
print("\nAll samples with multiple boundary records:")
print(low_records.groupby(['sample', 'haplotype']).size().sort_values(ascending=False))

# check locus length vs copy number slope for both methods
metadata_full = pd.read_csv("/Users/wagueacarinefongang/Desktop/MUC1_copy_numbers.tsv", sep="\t")
rm = pd.read_csv("/Users/wagueacarinefongang/Desktop/MUC1_complete_summary.tsv", sep="\t")
df_check = df_valid.merge(rm[['sample', 'haplotype', 'total_length']], on=['sample', 'haplotype'])

slope_jf, _, _, _, _ = stats.linregress(df_check['copy_number'], df_check['total_length'])
slope_trf, _, _, _, _ = stats.linregress(df_check['trf_copy_number'], df_check['total_length'])

print(f"\nBases per copy - Jellyfish: {slope_jf:.1f} bp")
print(f"Bases per copy - TRF: {slope_trf:.1f} bp")
print(f"Known MUC1 repeat unit: 60 bp")
print(f"Closer to 60: {'Jellyfish' if abs(slope_jf-60) < abs(slope_trf-60) else 'TRF'}")


# save TRF copy numbers for use in population figures
df_valid_out = df_valid[['sample', 'haplotype', 'continental_group', 
                          'population', 'trf_copy_number']].copy()
df_valid_out.rename(columns={'trf_copy_number': 'copy_number'}, inplace=True)
df_valid_out['method'] = 'TRF'
df_valid_out.to_csv("/Users/wagueacarinefongang/Desktop/MUC1_TRF_copy_numbers.tsv", 
                    sep="\t", index=False)
print("TRF copy numbers saved!")
print(f"Total samples: {df_valid_out.shape[0]}")
print(f"EAS median: {df_valid_out[df_valid_out['continental_group']=='EAS']['copy_number'].median():.1f}")