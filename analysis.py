import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# load data
df = pd.read_csv("/Users/wagueacarinefongang/Desktop/MUC1_copy_numbers.tsv", sep="\t")

# remove reference
df = df[df['continental_group'] != 'REF']

# basic summary
print("Dataset shape:", df.shape)
print("\nSamples per population:")
print(df['continental_group'].value_counts())
print("\nCopy number statistics by population:")
print(df.groupby('continental_group')['copy_number'].describe().to_string())

# statistical tests
afr = df[df['continental_group'] == 'AFR']['copy_number']
eas = df[df['continental_group'] == 'EAS']['copy_number']
eur = df[df['continental_group'] == 'EUR']['copy_number']
amr = df[df['continental_group'] == 'AMR']['copy_number']
sas = df[df['continental_group'] == 'SAS']['copy_number']

stat, pvalue = stats.kruskal(afr, eas, eur, amr, sas)
print(f"\nKruskal-Wallis test: H={stat:.3f}, p={pvalue:.6f}")

print("\nPairwise Mann-Whitney U tests:")
groups = {'AFR': afr, 'EAS': eas, 'EUR': eur, 'AMR': amr, 'SAS': sas}
for name1, g1 in groups.items():
    for name2, g2 in groups.items():
        if name1 < name2:
            stat, p = stats.mannwhitneyu(g1, g2)
            if p < 0.05:
                sig = "Significant"
            elif p < 0.10:
                sig = "Borderline"
            else:
                sig = "Not significant"
            print(f"  {name1} vs {name2}: p={p:.4f} {sig}")

# figures
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

sns.boxplot(data=df, x="continental_group", y="copy_number",
            order=["AFR", "EUR", "EAS", "AMR", "SAS"],
            hue="continental_group", legend=False,
            palette="Set2", ax=axes[0])
sns.stripplot(data=df, x="continental_group", y="copy_number",
              order=["AFR", "EUR", "EAS", "AMR", "SAS"],
              color="black", alpha=0.4, size=4, ax=axes[0])
axes[0].set_title("Box Plot")
axes[0].set_xlabel("Continental Group")
axes[0].set_ylabel("Estimated Copy Number")

sns.violinplot(data=df, x="continental_group", y="copy_number",
               order=["AFR", "EUR", "EAS", "AMR", "SAS"],
               hue="continental_group", legend=False,
               palette="Set2", ax=axes[1])
axes[1].set_title("Violin Plot")
axes[1].set_xlabel("Continental Group")
axes[1].set_ylabel("Estimated Copy Number")

plt.suptitle(
    "MUC1 VNTR Copy Number Across Global Populations\n(n=134 haplotypes)",
    fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()