import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re

# path to repeatmasker output
rm_dir = "/Users/wagueacarinefongang/Desktop/repeatmasker"

# parse each .tbl file
results = []

for fname in os.listdir(rm_dir):
    if not fname.endswith(".tbl"):
        continue
    
    parts = fname.split("_hap")
    sample = parts[0]
    haplotype = "hap" + parts[1].split(".")[0]
    
    fpath = os.path.join(rm_dir, fname)
    with open(fpath) as f:
        content = f.read()
    
    def extract(pattern, text):
        match = re.search(pattern, text)
        return float(match.group(1)) if match else 0.0
    
    total_length = extract(r"total length:\s+([\d]+) bp", content)
    bases_masked = extract(r"bases masked:\s+([\d]+) bp", content)
    sines = extract(r"SINEs:\s+\d+\s+([\d]+) bp", content)
    alus = extract(r"ALUs\s+\d+\s+([\d]+) bp", content)
    lines = extract(r"LINEs:\s+\d+\s+([\d]+) bp", content)
    ltr = extract(r"LTR elements:\s+\d+\s+([\d]+) bp", content)
    simple = extract(r"Simple repeats:\s+\d+\s+([\d]+) bp", content)
    
    results.append({
        "sample": sample,
        "haplotype": haplotype,
        "total_length": total_length,
        "bases_masked": bases_masked,
        "pct_masked": bases_masked / total_length * 100 if total_length > 0 else 0,
        "SINE_bp": sines,
        "ALU_bp": alus,
        "LINE_bp": lines,
        "LTR_bp": ltr,
        "simple_repeat_bp": simple
    })

df_rm = pd.DataFrame(results)

# merge with population metadata
metadata = pd.read_csv("/Users/wagueacarinefongang/Desktop/MUC1_copy_numbers.tsv", sep="\t")
df_rm = df_rm.merge(metadata[["sample", "haplotype", "continental_group", "population"]], 
                     on=["sample", "haplotype"], how="left")

print("Shape:", df_rm.shape)
print("\nSamples per population:")
print(df_rm["continental_group"].value_counts())
print("\nPercent masked by population:")
print(df_rm.groupby("continental_group")["pct_masked"].describe().round(2))
print(df_rm.groupby("continental_group")["pct_masked"].describe().round(2).to_string())



# remove REF
df_plot = df_rm[df_rm['continental_group'] != 'REF']

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# percent masked
sns.boxplot(data=df_plot, x="continental_group", y="pct_masked",
            order=["AFR", "EUR", "EAS", "AMR", "SAS"],
            hue="continental_group", legend=False,
            palette="Set2", ax=axes[0])
axes[0].set_title("Percent Masked")
axes[0].set_xlabel("Population")
axes[0].set_ylabel("% sequence masked")

# ALU content
sns.boxplot(data=df_plot, x="continental_group", y="ALU_bp",
            order=["AFR", "EUR", "EAS", "AMR", "SAS"],
            hue="continental_group", legend=False,
            palette="Set2", ax=axes[1])
axes[1].set_title("Alu Element Content")
axes[1].set_xlabel("Population")
axes[1].set_ylabel("Alu bases (bp)")

# LINE content
sns.boxplot(data=df_plot, x="continental_group", y="LINE_bp",
            order=["AFR", "EUR", "EAS", "AMR", "SAS"],
            hue="continental_group", legend=False,
            palette="Set2", ax=axes[2])
axes[2].set_title("LINE Element Content")
axes[2].set_xlabel("Population")
axes[2].set_ylabel("LINE bases (bp)")

plt.suptitle("MUC1 Locus Repeat Content Across Global Populations",
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig("/Users/wagueacarinefongang/Desktop/MUC1_repeatmasker.png", dpi=150)
plt.show()
print("Plot saved!")

# find the outlier sample
outlier = df_rm[df_rm['ALU_bp'] > 32000][['sample', 'haplotype', 'ALU_bp', 'LINE_bp', 'total_length', 'continental_group']]
print("High ALU outlier:")
print(outlier)

low_outliers = df_rm[df_rm['ALU_bp'] < 29000][['sample', 'haplotype', 'ALU_bp', 'LINE_bp', 'total_length', 'continental_group']]
print("\nLow ALU outliers:")
print(low_outliers)
print("High ALU outlier:")
print(outlier.to_string())


# checking which sample is the outlier with high ALU content
cn = df_rm[df_rm['sample'] == 'NA19331'][['sample', 'haplotype', 'total_length', 'ALU_bp']]
print("NA19331 locus info:")
print(cn)

# now checking its MUC1 copy number from my Jellyfish results
# to see if the longer locus matches a higher copy number
cn2 = metadata[metadata['sample'] == 'NA19331'][['sample', 'haplotype', 'copy_number', 'continental_group']]
print("\nNA19331 copy number:")
print(cn2)

# checking full details of NA19331 both haplotypes
print("NA19331 locus info:")
print(df_rm[df_rm['sample'] == 'NA19331'][['sample', 'haplotype', 'total_length', 'ALU_bp', 'LINE_bp']].to_string())

# now checking copy number for both haplotypes
print("\nNA19331 copy number:")
print(metadata[metadata['sample'] == 'NA19331'][['sample', 'haplotype', 'copy_number']].to_string())


# flagging NA19331 hap1 as an outlier
# it has a longer total locus length (146,778 bp vs average 126,000 bp)
# driven by extra Alu (35,035 bp vs average 29,550 bp)
# and extra LINE (15,261 bp vs average 12,900 bp) content
# BUT has lower VNTR copy number (28.10) compared to hap2 (46.08)
# suggests extra mobile element insertions in flanking regions
# not directly related to VNTR copy number variation
# worth flagging to Sharon for further investigation
print("Flagging NA19331 hap1 as outlier - longer locus with extra mobile elements")


# merge copy number with repeat masker results
df_combined = df_rm.merge(
    metadata[["sample", "haplotype", "copy_number"]], 
    on=["sample", "haplotype"]
)

# is there a correlation between copy number and total locus length?
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_combined[df_combined['continental_group'] != 'REF'],
                x="copy_number", y="total_length",
                hue="continental_group",
                palette="Set2", alpha=0.7)
plt.title("MUC1 Copy Number vs Total Locus Length")
plt.xlabel("Estimated Copy Number")
plt.ylabel("Total Locus Length (bp)")
plt.savefig("/Users/wagueacarinefongang/Desktop/MUC1_copynumber_vs_length.png", dpi=150)
plt.show()

# creating my final summary table combining all results
df_final = df_rm.merge(
    metadata[["sample", "haplotype", "copy_number"]], 
    on=["sample", "haplotype"]
)

# selecting the key columns for the summary
df_summary = df_final[[
    "sample", "haplotype", "continental_group", "population",
    "copy_number", "total_length", "pct_masked", 
    "ALU_bp", "LINE_bp", "simple_repeat_bp"
]].copy()

# rounding numbers for readability
df_summary["copy_number"] = df_summary["copy_number"].round(2)
df_summary["pct_mas./ked"] = df_summary["pct_masked"].round(2)

# saving to desktop
df_summary.to_csv("/Users/wagueacarinefongang/Desktop/MUC1_complete_summary.tsv", 
                  sep="\t", index=False)
print("Summary table saved!")
print(df_summary.shape)
print(df_summary.head())


from scipy import stats

# RepeatMasker statistics per population
print("\nPercent masked full statistics:")
print(df_rm[df_rm['continental_group'] != 'REF'].groupby('continental_group')['pct_masked'].describe().round(2).to_string())

# correlation between copy number and locus length
df_combined = df_rm.merge(metadata[["sample", "haplotype", "copy_number"]], on=["sample", "haplotype"])
df_combined = df_combined[df_combined['continental_group'] != 'REF']
r, p = stats.pearsonr(df_combined['copy_number'], df_combined['total_length'])
print(f"\nCorrelation copy number vs locus length: r={r:.3f}, p={p:.6f}")