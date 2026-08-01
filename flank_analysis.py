import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# ─── Configuration ────────────────────────────────────────────────────────────

FLANK_PATH = "/Users/wagueacarinefongang/Desktop/flank_anchoring_results.tsv"
JELLY_PATH = "/Users/wagueacarinefongang/Desktop/MUC1_copy_numbers.tsv"
TRF_PATH = "/Users/wagueacarinefongang/Desktop/MUC1_TRF_copy_numbers.tsv"
OUTPUT = "/Users/wagueacarinefongang/Desktop/MUC1_flank_validation.png"

# ─── Load all three measurements ──────────────────────────────────────────────

# flank anchoring measures the array by aligning 1kb of unique sequence from
# each side of the VNTR to the assembly and taking the distance between where
# the two anchors land. it never counts k-mers and never looks for repeats,
# so it is independent of both Jellyfish and TRF
flank = pd.read_csv(FLANK_PATH, sep="\t")
flank = flank[['sample', 'haplotype', 'array_bp', 'copy_number']]
flank.columns = ['sample', 'haplotype', 'array_bp', 'flank_copies']

jelly = pd.read_csv(JELLY_PATH, sep="\t")

trf = pd.read_csv(TRF_PATH, sep="\t")
trf = trf[['sample', 'haplotype', 'copy_number']]
trf.columns = ['sample', 'haplotype', 'trf_copies']

# joining all three so every row has the same haplotype measured three ways
df = flank.merge(jelly, on=['sample', 'haplotype']).merge(trf, on=['sample', 'haplotype'])
df = df[df['continental_group'] != 'REF']

print(f"Haplotypes with all three measurements: {len(df)}")

# ─── Compare flank anchoring to each method ───────────────────────────────────

# if flank anchoring agrees with one method much more closely than the other,
# that tells me which one is calibrated correctly
for name, col in [("Jellyfish", "copy_number"), ("TRF", "trf_copies")]:
    r, p = stats.pearsonr(df['flank_copies'], df[col])
    slope, intercept, _, _, _ = stats.linregress(df[col], df['flank_copies'])
    ratio = (df['flank_copies'] / df[col]).mean()
    print(f"\nFlank vs {name}:")
    print(f"  r = {r:.4f}")
    print(f"  slope = {slope:.3f}, intercept = {intercept:.3f}")
    print(f"  mean ratio (flank/{name}) = {ratio:.3f}")

# ─── Which method is closer ───────────────────────────────────────────────────

# a slope near 1 means the two methods are on the same scale
# the mean absolute difference says how far apart they are in actual copies
print("\nMean absolute difference from flank anchoring:")
print(f"  Jellyfish: {(df['flank_copies'] - df['copy_number']).abs().mean():.2f} copies")
print(f"  TRF:       {(df['flank_copies'] - df['trf_copies']).abs().mean():.2f} copies")

# ─── Plot ─────────────────────────────────────────────────────────────────────

# two panels so the comparison is direct. the dashed line is where points would
# fall if the two methods agreed exactly
fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))

for ax, col, label in zip(axes, ['copy_number', 'trf_copies'], ['Jellyfish', 'TRF']):
    sns.scatterplot(data=df, x=col, y='flank_copies',
                    hue='continental_group', palette='Set2',
                    alpha=0.7, ax=ax)

    lim = max(df['flank_copies'].max(), df[col].max()) * 1.05
    ax.plot([0, lim], [0, lim], 'k--', alpha=0.4, label='perfect agreement')

    r, _ = stats.pearsonr(df['flank_copies'], df[col])
    ax.set_xlabel(f"{label} copy number")
    ax.set_ylabel("Flank-anchored copy number")
    ax.set_title(f"Flank anchoring vs {label}  (r = {r:.3f})")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.legend(fontsize=8)

plt.suptitle("Independent validation: array length measured by flank anchoring",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved.")