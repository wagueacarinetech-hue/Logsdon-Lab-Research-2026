import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from scipy.stats import mannwhitneyu, fisher_exact, chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import proportion_confint
import numpy as np

# ─── Configuration ────────────────────────────────────────────────────────────

DATA_PATH = "/Users/wagueacarinefongang/Desktop/MUC1_copy_numbers.tsv"
OUTPUT_DIR = "/Users/wagueacarinefongang/Desktop"

# the MUC1 VNTR copy number distribution is bimodal
# there are two common allele lengths — short (~29 copies) and long (~55 copies)
# the valley between the two peaks falls around 40 copies
# this cutoff separates the two allele classes
CUTOFF = 40

# population order for all plots
# and a colorblind-friendly palette
ORDER = ["AFR", "AMR", "EAS", "EUR", "SAS"]
PALETTE = {
    "AFR": "#C0392B",
    "AMR": "#8E44AD",
    "EAS": "#2980B9",
    "EUR": "#27AE60",
    "SAS": "#E67E22"
}

# ─── Load data ────────────────────────────────────────────────────────────────

# jellyfish copy number estimates for all 139 haplotypes
# CHM13 reference removed — only population samples used
df = pd.read_csv(DATA_PATH, sep="\t")
df = df[df['continental_group'] != 'REF']

# ─── Basic summary ────────────────────────────────────────────────────────────

print("Dataset shape:", df.shape)
print("\nSamples per population:")
print(df['continental_group'].value_counts())
print("\nCopy number statistics by population:")
print(df.groupby('continental_group')['copy_number'].describe().round(2).to_string())

# ─── Kruskal-Wallis test ──────────────────────────────────────────────────────

# testing whether copy number differs significantly across all 5 populations
# using Kruskal-Wallis because the data is not normally distributed
# this is the omnibus test — if significant it justifies pairwise comparisons
afr = df[df['continental_group'] == 'AFR']['copy_number']
eas = df[df['continental_group'] == 'EAS']['copy_number']
eur = df[df['continental_group'] == 'EUR']['copy_number']
amr = df[df['continental_group'] == 'AMR']['copy_number']
sas = df[df['continental_group'] == 'SAS']['copy_number']

stat, pvalue = stats.kruskal(afr, eas, eur, amr, sas)
print(f"\nKruskal-Wallis: H={stat:.3f}, p={pvalue:.6f}")

# ─── Pairwise tests with Bonferroni correction ────────────────────────────────

# all 10 pairwise comparisons between the 5 populations
# Bonferroni correction applied to control for multiple testing
# with 10 tests at alpha=0.05 we would expect 0.5 false positives by chance
# correction makes the threshold stricter: 0.05/10 = 0.005 per test
pvalues = []
pairs = []
groups_list = ['AFR', 'AMR', 'EAS', 'EUR', 'SAS']

for i in range(len(groups_list)):
    for j in range(i+1, len(groups_list)):
        g1 = groups_list[i]
        g2 = groups_list[j]
        s1 = df[df['continental_group']==g1]['copy_number']
        s2 = df[df['continental_group']==g2]['copy_number']
        _, p = mannwhitneyu(s1, s2)
        pvalues.append(p)
        pairs.append((g1, g2))

reject, pvals_corrected, _, _ = multipletests(pvalues, method='bonferroni')

print("\nPairwise tests with Bonferroni correction:")
for i, (g1, g2) in enumerate(pairs):
    sig = "SIGNIFICANT" if reject[i] else "not significant"
    print(f"  {g1} vs {g2}: p_raw={pvalues[i]:.4f} p_corrected={pvals_corrected[i]:.4f} {sig}")

# only keeping pairs that survived correction for plotting
sig_pairs = [(g1, g2, pvals_corrected[i])
             for i, (g1, g2) in enumerate(pairs) if reject[i]]
print(f"\nSignificant pairs after Bonferroni: {len(sig_pairs)}")

# ─── AFR vs non-AFR continuous comparison ─────────────────────────────────────

# collapsing all non-African populations into one group
# this is the main clinical question — do African populations differ
# from all others in MUC1 copy number?
df['group'] = df['continental_group'].apply(
    lambda x: 'AFR' if x == 'AFR' else 'non-AFR'
)

afr_group = df[df['group'] == 'AFR']['copy_number']
nonafr_group = df[df['group'] == 'non-AFR']['copy_number']

_, p_afr = mannwhitneyu(afr_group, nonafr_group)
print(f"\nAFR vs non-AFR continuous: p={p_afr:.4f}")
print(f"AFR median: {afr_group.median():.2f}")
print(f"non-AFR median: {nonafr_group.median():.2f}")
print("SIGNIFICANT" if p_afr < 0.05 else "Not significant")

# ─── Allele frequency analysis ────────────────────────────────────────────────

# classifying each haplotype as carrying the short or long allele
# the bimodal distribution has two peaks at ~29 and ~55 copies
# the valley between them falls at ~40 copies which is our cutoff
# using >= CUTOFF for long to avoid ambiguity at exactly 40 copies
df['allele_type'] = df['copy_number'].apply(
    lambda x: 'long (>=40)' if x >= CUTOFF else 'short (<40)'
)

print("\nAllele type counts:")
print(df['allele_type'].value_counts())

# building frequency table per population
freq_table = df.groupby(
    ['continental_group', 'allele_type']
).size().unstack(fill_value=0)

freq_table['total'] = freq_table.sum(axis=1)
freq_table['pct_short'] = (
    freq_table['short (<40)'] / freq_table['total'] * 100
).round(1)
freq_table['pct_long'] = (
    freq_table['long (>=40)'] / freq_table['total'] * 100
).round(1)

print("\nAllele frequencies by population:")
print(freq_table.to_string())

# chi-square test on allele frequencies across all 5 populations
# testing whether which allele you carry depends on your ancestry
contingency = freq_table[['short (<40)', 'long (>=40)']].loc[ORDER]
chi2_stat, p_chi, dof, _ = chi2_contingency(contingency)
print(f"\nChi-square: chi2={chi2_stat:.3f}, p={p_chi:.6f}, df={dof}")

# ─── AFR vs non-AFR allele frequency ─────────────────────────────────────────

# computing non-AFR counts by summing all non-African populations
# from the frequency table — single source of truth
nonafr_counts = freq_table.drop('AFR')[['short (<40)', 'long (>=40)']].sum()
nonafr_short_n = int(nonafr_counts['short (<40)'])
nonafr_long_n = int(nonafr_counts['long (>=40)'])
nonafr_total = nonafr_short_n + nonafr_long_n

afr_short_n = int(freq_table.loc['AFR', 'short (<40)'])
afr_long_n = int(freq_table.loc['AFR', 'long (>=40)'])
afr_total = afr_short_n + afr_long_n

afr_short_pct = afr_short_n / afr_total * 100
afr_long_pct = afr_long_n / afr_total * 100
nonafr_short_pct = nonafr_short_n / nonafr_total * 100
nonafr_long_pct = nonafr_long_n / nonafr_total * 100

# Fisher's exact test is more appropriate than chi-square for 2x2 tables
# and does not assume large expected cell counts
odds_ratio, p_fisher = fisher_exact(
    [[afr_short_n, afr_long_n],
     [nonafr_short_n, nonafr_long_n]]
)

print(f"\nFisher's exact test — AFR vs non-AFR allele frequency:")
print(f"AFR: {afr_short_n} short ({afr_short_pct:.1f}%), {afr_long_n} long ({afr_long_pct:.1f}%)")
print(f"non-AFR: {nonafr_short_n} short ({nonafr_short_pct:.1f}%), {nonafr_long_n} long ({nonafr_long_pct:.1f}%)")
print(f"Odds ratio: {odds_ratio:.3f}")
print(f"p-value: {p_fisher:.6f}")
print("SIGNIFICANT" if p_fisher < 0.05 else "Not significant")

# ─── Wilson confidence intervals ──────────────────────────────────────────────

# Wilson method is preferred over normal approximation for proportions
# especially with small sample sizes like EUR and SAS (n=16)
# the CI width shows how much uncertainty there is in each estimate
print("\nWilson confidence intervals on long-allele frequency:")
print(f"{'Population':<12} {'n':<6} {'% long':<10} {'95% CI'}")
print("-" * 45)

long_pcts = []
long_cis_lo = []
long_cis_hi = []

for g in ORDER:
    n_total = freq_table.loc[g, 'total']
    n_long = freq_table.loc[g, 'long (>=40)']
    pct = n_long / n_total * 100
    lo, hi = proportion_confint(n_long, n_total, method='wilson')
    long_pcts.append(pct)
    long_cis_lo.append(pct - lo*100)
    long_cis_hi.append(hi*100 - pct)
    print(f"{g:<12} {n_total:<6} {pct:<10.1f} ({lo*100:.1f}% - {hi*100:.1f}%)")

short_pcts = [100 - p for p in long_pcts]

# ─── Sensitivity check — cutoff robustness ────────────────────────────────────

# testing whether the AFR vs non-AFR result holds
# across different choices of the short/long allele cutoff
# a result that holds across a range of cutoffs is robust
# a result that only appears at one specific cutoff is suspect
print("\nSensitivity check — AFR vs non-AFR Fisher p at different cutoffs:")
print(f"{'Cutoff':<10} {'AFR % long':<15} {'non-AFR % long':<18} {'p-value':<12} {'Significant'}")
print("-" * 65)

for cutoff in [35, 37, 38, 39, 40, 41, 42, 43, 45]:
    df['allele_tmp'] = df['copy_number'].apply(
        lambda x: 'long' if x >= cutoff else 'short'
    )
    a_s = len(df[(df['continental_group']=='AFR') & (df['allele_tmp']=='short')])
    a_l = len(df[(df['continental_group']=='AFR') & (df['allele_tmp']=='long')])
    na = df[df['continental_group']!='AFR']
    na_s = len(na[na['allele_tmp']=='short'])
    na_l = len(na[na['allele_tmp']=='long'])
    _, p = fisher_exact([[a_s, a_l], [na_s, na_l]])
    a_pct = a_l / (a_s + a_l) * 100
    na_pct = na_l / (na_s + na_l) * 100
    sig = "YES" if p < 0.05 else "no"
    print(f"{cutoff:<10} {a_pct:<15.1f} {na_pct:<18.1f} {p:<12.4f} {sig}")

# ─── Leave-one-out sensitivity ────────────────────────────────────────────────

# testing how much each population contributes to the overall chi-square signal
# by removing one group at a time
# if the result collapses when one group is removed that group is driving the signal
# this is transparent and defensible — the opposite of cherry picking
print("\nLeave-one-out chi-square sensitivity analysis:")
print(f"{'Removed':<20} {'chi2':<10} {'p-value':<12} {'Significant'}")
print("-" * 55)

for remove in ORDER:
    subset = contingency.drop(remove)
    c, p_loo, _, _ = chi2_contingency(subset)
    sig = "YES" if p_loo < 0.05 else "no"
    print(f"Remove {remove:<13} {c:<10.3f} {p_loo:<12.4f} {sig}")

c_all, p_all, _, _ = chi2_contingency(contingency)
print(f"{'None (all groups)':<20} {c_all:<10.3f} {p_all:<12.4f} "
      f"{'YES' if p_all < 0.05 else 'no'}")

# ─── Helper functions ─────────────────────────────────────────────────────────

def get_stars(p):
    """convert p-value to significance stars for plot annotations"""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return "ns"

def add_significance_bars(ax, sig_pairs, order, y_max, y_step=7):
    """draw significance brackets between pairs on a violin/box plot"""
    for i, (g1, g2, p) in enumerate(sig_pairs):
        x1 = order.index(g1)
        x2 = order.index(g2)
        y = y_max + y_step * (i + 1)
        ax.plot([x1, x1, x2, x2], [y-1, y, y, y-1],
                color='black', linewidth=1.5)
        ax.text((x1 + x2) / 2, y + 0.3, get_stars(p),
                ha='center', va='bottom', fontsize=13, fontweight='bold')
    return y_max + y_step * (len(sig_pairs) + 2)

# ─── Figure 1: Main violin plot — all 5 populations ──────────────────────────

fig, ax = plt.subplots(figsize=(13, 8))

# x axis labels include sample sizes
labels = [f"{g}\n(n={len(df[df['continental_group']==g])})" for g in ORDER]

# layering violin + boxplot + strip for maximum information
sns.violinplot(data=df, x="continental_group", y="copy_number",
               order=ORDER, hue="continental_group",
               palette=PALETTE, legend=False,
               ax=ax, inner=None, alpha=0.6)
sns.boxplot(data=df, x="continental_group", y="copy_number",
            order=ORDER, hue="continental_group",
            palette=PALETTE, legend=False,
            ax=ax, width=0.15, fliersize=0,
            boxprops=dict(alpha=0.8))
sns.stripplot(data=df, x="continental_group", y="copy_number",
              order=ORDER, hue="continental_group",
              palette=PALETTE, legend=False,
              ax=ax, size=3, alpha=0.5, jitter=True)

ax.set_xticks(range(len(ORDER)))
ax.set_xticklabels(labels, fontsize=11)

y_max = df['copy_number'].max()
y_lim = add_significance_bars(ax, sig_pairs, ORDER, y_max)

legend_patches = [mpatches.Patch(color=PALETTE[g], label=g) for g in ORDER]
ax.legend(handles=legend_patches, title="Continental group",
          loc='upper right', framealpha=0.9)

ax.set_title(f"MUC1 VNTR Copy Number Across Global Populations\n"
             f"Kruskal-Wallis p={pvalue:.4f} | Bonferroni corrected | "
             f"* p<0.05  ** p<0.01  *** p<0.001",
             fontsize=11, fontweight='bold')
ax.set_xlabel("")
ax.set_ylabel("Estimated Copy Number (Jellyfish)", fontsize=12)
ax.set_ylim(0, y_lim)

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/MUC1_figure1_populations.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 1 saved!")

# ─── Figure 2: AFR vs non-AFR ─────────────────────────────────────────────────

# two panel figure: left shows two groups, right shows all 5 colored by AFR status
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
order2 = ['AFR', 'non-AFR']
n_afr = len(afr_group)
n_nonafr = len(nonafr_group)
labels2 = [f"AFR\n(n={n_afr})", f"non-AFR\n(n={n_nonafr})"]
palette2 = {'AFR': '#C0392B', 'non-AFR': '#2980B9'}

sns.violinplot(data=df, x='group', y='copy_number',
               order=order2, hue='group',
               palette=palette2, legend=False,
               ax=axes[0], inner=None, alpha=0.6)
sns.boxplot(data=df, x='group', y='copy_number',
            order=order2, hue='group',
            palette=palette2, legend=False,
            ax=axes[0], width=0.15, fliersize=0)
sns.stripplot(data=df, x='group', y='copy_number',
              order=order2, hue='group',
              palette=palette2, legend=False,
              ax=axes[0], size=3, alpha=0.5, jitter=True)

axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(labels2, fontsize=12)

y_top = df['copy_number'].max() + 5
axes[0].plot([0, 0, 1, 1], [y_top-2, y_top, y_top, y_top-2],
             color='black', linewidth=1.5)
axes[0].text(0.5, y_top + 0.5,
             f"p={p_afr:.4f} ({get_stars(p_afr)})",
             ha='center', fontsize=10)
axes[0].set_ylim(0, y_top + 10)
axes[0].set_title("AFR vs non-AFR", fontsize=12, fontweight='bold')
axes[0].set_xlabel("")
axes[0].set_ylabel("Estimated Copy Number (Jellyfish)")

# right panel colors AFR red and all others blue
palette_afr = {g: '#C0392B' if g == 'AFR' else '#2980B9' for g in ORDER}

sns.violinplot(data=df, x="continental_group", y="copy_number",
               order=ORDER, hue="continental_group",
               palette=palette_afr, legend=False,
               ax=axes[1], inner=None, alpha=0.6)
sns.boxplot(data=df, x="continental_group", y="copy_number",
            order=ORDER, hue="continental_group",
            palette=palette_afr, legend=False,
            ax=axes[1], width=0.15, fliersize=0)
sns.stripplot(data=df, x="continental_group", y="copy_number",
              order=ORDER, hue="continental_group",
              palette=palette_afr, legend=False,
              ax=axes[1], size=3, alpha=0.5, jitter=True)

labels_afr = [f"{g}\n(n={len(df[df['continental_group']==g])})" for g in ORDER]
axes[1].set_xticks(range(len(ORDER)))
axes[1].set_xticklabels(labels_afr, fontsize=10)
axes[1].set_title("All populations (red=AFR, blue=non-AFR)",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("")
axes[1].set_ylabel("Estimated Copy Number (Jellyfish)")

legend_patches2 = [
    mpatches.Patch(color='#C0392B', label='African (AFR)'),
    mpatches.Patch(color='#2980B9', label='non-African')
]
axes[1].legend(handles=legend_patches2, loc='upper right', framealpha=0.9)

plt.suptitle("MUC1 VNTR Copy Number: African vs non-African Populations",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/MUC1_figure2_AFR_vs_nonAFR.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 2 saved!")

# ─── Figure 3: Bimodal distribution ──────────────────────────────────────────

# showing the two-humped distribution that motivates the short/long allele framing
# left panel shows all haplotypes together
# right panel breaks it down by population
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

axes[0].hist(df['copy_number'], bins=30, color='steelblue',
             edgecolor='white', alpha=0.8)
axes[0].axvline(x=29, color='red', linestyle='--', linewidth=1.5,
                label='Short allele peak (~29 copies)')
axes[0].axvline(x=55, color='orange', linestyle='--', linewidth=1.5,
                label='Long allele peak (~55 copies)')
axes[0].set_title("Distribution of MUC1 Copy Number\n(all haplotypes)",
                  fontsize=12, fontweight='bold')
axes[0].set_xlabel("Estimated Copy Number (Jellyfish)")
axes[0].set_ylabel("Number of haplotypes")
axes[0].legend()

for g in ORDER:
    subset = df[df['continental_group']==g]['copy_number']
    axes[1].hist(subset, bins=20, alpha=0.5,
                label=f"{g} (n={len(subset)})",
                color=PALETTE[g], edgecolor='white')

axes[1].axvline(x=29, color='red', linestyle='--', linewidth=1.5)
axes[1].axvline(x=55, color='orange', linestyle='--', linewidth=1.5)
axes[1].set_title("Distribution by Population",
                  fontsize=12, fontweight='bold')
axes[1].set_xlabel("Estimated Copy Number (Jellyfish)")
axes[1].set_ylabel("Number of haplotypes")
axes[1].legend(fontsize=9)

plt.suptitle("Bimodal Distribution of MUC1 VNTR Copy Number",
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/MUC1_figure3_bimodal.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 3 saved!")

# ─── Figure 4: P-value table ──────────────────────────────────────────────────

# visual table showing all pairwise comparisons
# significant rows highlighted in red
fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

table_data = []
for i, (g1, g2) in enumerate(pairs):
    table_data.append([
        f"{g1} vs {g2}",
        f"{pvalues[i]:.4f}",
        f"{pvals_corrected[i]:.4f}",
        "significant" if reject[i] else "not significant"
    ])

table = ax.table(
    cellText=table_data,
    colLabels=["Comparison", "p-value (raw)",
               "p-value (Bonferroni)", "Result"],
    cellLoc='center', loc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.8)

for i in range(len(pairs)):
    if reject[i]:
        for j in range(4):
            table[(i+1, j)].set_facecolor('#FADBD8')

plt.title("Pairwise Statistical Comparisons\n"
          "(Mann-Whitney U test, Bonferroni corrected)",
          fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/MUC1_figure4_pvalue_table.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 4 saved!")

# ─── Figure 5: Allele frequency with confidence intervals ────────────────────

# stacked bar chart showing short vs long allele proportion per population
# error bars show 95% Wilson confidence intervals on the long-allele proportion
# placed at the short/long boundary where the uncertainty actually lives
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
x = np.arange(len(ORDER))

axes[0].bar(x, short_pcts, color='#2980B9', alpha=0.8,
            label='Short allele (<40 copies)', edgecolor='white')
axes[0].bar(x, long_pcts, bottom=short_pcts, color='#E74C3C',
            alpha=0.8, label='Long allele (>=40 copies)', edgecolor='white')

# error bars at the short/long boundary — where the uncertainty lives
axes[0].errorbar(x, short_pcts,
                yerr=[long_cis_lo, long_cis_hi],
                fmt='none', color='black', capsize=5, linewidth=1.5)

# percentage labels inside bars
for i, (s, l) in enumerate(zip(short_pcts, long_pcts)):
    if s > 10:
        axes[0].text(i, s/2, f"{s:.0f}%",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')
    if l > 10:
        axes[0].text(i, s + l/2, f"{l:.0f}%",
                    ha='center', va='center',
                    fontsize=10, fontweight='bold', color='white')

# sample sizes below bars
for i, g in enumerate(ORDER):
    n = freq_table.loc[g, 'total']
    axes[0].text(i, -5, f"n={n}", ha='center', fontsize=9)

axes[0].set_xticks(x)
axes[0].set_xticklabels(ORDER, fontsize=11)
axes[0].set_ylim(-8, 115)
axes[0].set_title("Short vs Long Allele Frequency by Population\n"
                  "(error bars = 95% Wilson CI on long-allele proportion)",
                  fontsize=11, fontweight='bold')
axes[0].set_xlabel("")
axes[0].set_ylabel("Percentage of haplotypes (%)")
axes[0].legend(loc='upper right', fontsize=9)
axes[0].text(len(ORDER)/2 - 0.5, 108,
             f"Chi-square p={p_chi:.4f}",
             ha='center', fontsize=10, color='red')

# right panel — AFR vs non-AFR with Wilson CI
afr_lo_ci, afr_hi_ci = proportion_confint(
    afr_long_n, afr_total, method='wilson')
na_lo_ci, na_hi_ci = proportion_confint(
    nonafr_long_n, nonafr_total, method='wilson')

short_pcts2 = [afr_short_pct, nonafr_short_pct]
long_pcts2 = [afr_long_pct, nonafr_long_pct]
lo_errs = [afr_long_pct - afr_lo_ci*100,
           nonafr_long_pct - na_lo_ci*100]
hi_errs = [afr_hi_ci*100 - afr_long_pct,
           na_hi_ci*100 - nonafr_long_pct]

axes[1].bar([0, 1], short_pcts2, color='#2980B9', alpha=0.8,
            label='Short allele', edgecolor='white')
axes[1].bar([0, 1], long_pcts2, bottom=short_pcts2,
            color='#E74C3C', alpha=0.8,
            label='Long allele', edgecolor='white')
axes[1].errorbar([0, 1], short_pcts2,
                yerr=[lo_errs, hi_errs],
                fmt='none', color='black', capsize=5, linewidth=1.5)

for i, (s, l) in enumerate(zip(short_pcts2, long_pcts2)):
    axes[1].text(i, s/2, f"{s:.0f}%",
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')
    axes[1].text(i, s + l/2, f"{l:.0f}%",
                ha='center', va='center',
                fontsize=12, fontweight='bold', color='white')

axes[1].text(0, -5, f"n={afr_total}", ha='center', fontsize=9)
axes[1].text(1, -5, f"n={nonafr_total}", ha='center', fontsize=9)

# significance bar
axes[1].plot([0, 0, 1, 1], [107, 109, 109, 107],
             color='black', linewidth=1.5)
axes[1].text(0.5, 110,
             f"p={p_fisher:.4f} ({get_stars(p_fisher)})",
             ha='center', fontsize=10)

axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(
    [f"AFR\n(n={afr_total})", f"non-AFR\n(n={nonafr_total})"],
    fontsize=12)
axes[1].set_ylim(-8, 120)
axes[1].set_title("Short vs Long Allele Frequency\nAFR vs non-AFR",
                  fontsize=11, fontweight='bold')
axes[1].set_xlabel("")
axes[1].set_ylabel("Percentage of haplotypes (%)")
axes[1].legend(loc='upper right', fontsize=9)

plt.suptitle("MUC1 VNTR Allele Frequency Across Global Populations\n"
             "(Short allele <40 copies, Long allele >=40 copies | "
             "Jellyfish copy number estimates)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/MUC1_figure5_allele_frequency.png",
            dpi=150, bbox_inches='tight')
plt.show()
print("Figure 5 saved!")

print("\nAll figures saved!")
print("Figure 1: MUC1_figure1_populations.png")
print("Figure 2: MUC1_figure2_AFR_vs_nonAFR.png")
print("Figure 3: MUC1_figure3_bimodal.png")
print("Figure 4: MUC1_figure4_pvalue_table.png")
print("Figure 5: MUC1_figure5_allele_frequency.png")