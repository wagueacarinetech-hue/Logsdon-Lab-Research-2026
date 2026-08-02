#!/usr/bin/env python3
"""
RepeatContact: analysing 3D chromatin contacts within tandem repeat arrays
using repeat-unit coordinates.

Existing chromatin contact tools work in linear genomic coordinates. But
tandem repeat arrays like MUC1 have different lengths in different people, so
the same genomic offset means something different in each one. This tool
converts contacts into repeat-unit coordinates, which puts arrays of different
lengths on a comparable scale and makes it possible to ask whether they fold
the same way.

Author: Carine Waguea Fongang
Logsdon Lab, University of Pennsylvania
Summer 2026
"""

import numpy as np
import pandas as pd
from scipy import fft, signal as sig
import matplotlib.pyplot as plt


# ─── Part 1: Coordinate transformer ───────────────────────────────────────────

def genomic_to_repeat_unit(position, array_start, period):
    """
    Convert a genomic position into a repeat-unit coordinate.

    This is the core idea of the tool. Instead of asking what genomic position
    a contact is at, ask which repeat copy it falls in.

    If the array starts at 63108 and each copy is 60bp:
      63108 -> unit 0.0
      63168 -> unit 1.0
      63198 -> unit 1.5   (halfway through the second copy)

    Returns a float, so the fractional part tells you where within the copy the
    position falls. Returns None if the position is before the array start.
    """
    offset = position - array_start
    if offset < 0:
        return None
    return offset / period


def positions_to_units(positions, array_start, period):
    """
    Vectorised version of the above, for whole arrays of positions at once.

    Positions before the array start come back as NaN rather than None, so the
    result stays a numeric array and can be masked.
    """
    units = (np.asarray(positions, dtype=float) - array_start) / period
    units[units < 0] = np.nan
    return units


# ─── Part 2: Contact matrix builder ───────────────────────────────────────────

def build_contact_matrix(contacts_df, array_start, period, n_copies):
    """
    Build the copy-by-copy contact matrix.

    matrix[i][j] counts how often copy i was in contact with copy j. A strong
    diagonal means nearby copies touch often, which is expected. Off-diagonal
    structure is what would indicate higher-order folding.

    Two details worth knowing.

    The matrix is symmetric, since a contact between i and j is also a contact
    between j and i. But a self-contact where i equals j must only be counted
    once, otherwise the diagonal carries double weight relative to everything
    else.

    This is vectorised rather than looping over rows. On a few thousand
    contacts the difference is seconds against milliseconds; on a real dataset
    it is the difference between usable and not.
    """
    matrix = np.zeros((n_copies, n_copies))

    if len(contacts_df) == 0:
        return matrix

    u1 = positions_to_units(contacts_df['pos1'].values, array_start, period)
    u2 = positions_to_units(contacts_df['pos2'].values, array_start, period)

    # keep only contacts where both ends land inside the array
    ok = (~np.isnan(u1)) & (~np.isnan(u2)) & (u1 < n_copies) & (u2 < n_copies)
    i = u1[ok].astype(int)
    j = u2[ok].astype(int)

    # np.add.at accumulates repeated indices correctly, unlike matrix[i, j] += 1
    # which would only apply the last occurrence of any duplicated pair
    np.add.at(matrix, (i, j), 1)

    # mirror the off-diagonal entries, leaving the diagonal counted once
    off = i != j
    np.add.at(matrix, (j[off], i[off]), 1)

    return matrix


# ─── Part 3: Periodicity detection ────────────────────────────────────────────

def diagonal_profile(matrix):
    """
    Mean contact frequency at each separation distance.

    Reading down the diagonals of the contact matrix: offset 0 is self-contacts,
    offset 1 is neighbouring copies, offset 2 is copies two apart, and so on.
    This collapses the matrix into a one-dimensional signal.
    """
    n = matrix.shape[0]
    return np.array([np.mean(np.diagonal(matrix, offset=k)) for k in range(n)])


def fourier_analysis(matrix, detrend=True):
    """
    Look for periodic structure in how contact frequency varies with separation.

    A peak at period N would mean every Nth copy preferentially contacts every
    other Nth copy, which is what a regular folding pattern would produce.

    The detrending matters and is on by default. Contact frequency falls off
    steeply with separation, because that is what polymers do. That falloff is
    an enormous low-frequency component, and without removing it the transform
    reports the decay curve rather than anything sitting on top of it. The
    dominant frequency then comes out near the array length regardless of the
    input, which is not a finding about folding.

    Fitting the decay in log space and subtracting it leaves the residual,
    which is where a genuine periodic signal lives.

    Returns frequencies, power spectrum, and the signal that was transformed.
    """
    profile = diagonal_profile(matrix)
    n = len(profile)

    if not detrend:
        spectrum = np.abs(fft.fft(profile))
        freqs = fft.fftfreq(n)
        keep = freqs > 0
        return freqs[keep], spectrum[keep], profile

    # skip offset 0, which is self-contacts and not part of the decay
    x = np.arange(1, n)
    y = profile[1:]
    ok = y > 0

    if ok.sum() < 5:
        return np.array([]), np.array([]), profile

    coef = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    trend = np.exp(np.polyval(coef, np.log(x)))
    residual = y - trend

    # taper the ends so the transform does not see an artificial step at the
    # boundaries and report a spurious frequency from it
    windowed = residual * np.hanning(len(residual))

    spectrum = np.abs(fft.fft(windowed))
    freqs = fft.fftfreq(len(windowed))
    keep = freqs > 0

    return freqs[keep], spectrum[keep], residual


def dominant_period(freqs, spectrum, n_copies,
                    min_period=2.5, height_frac=0.2):
    """
    Report the folding period, if there is one.

    Taking the tallest peak in the spectrum gives the wrong answer, and this
    was a real bug caught by the sensitivity test. A signal with period 5 is
    not a pure sine wave, so it also produces power at 2.5, at 1.67 and so on.
    Those harmonics can be taller than the fundamental, and the problem gets
    worse with more data rather than better, so the tool reported exactly half
    the true period increasingly reliably as the input improved.

    Instead: find every local peak above a fraction of the maximum, then take
    the one with the longest period. Harmonics always sit at shorter periods
    than the fundamental that produced them.

    Returns NaN when no peak clears the threshold, which is the correct answer
    for data with no periodic structure. The tool should decline rather than
    invent a number.
    """
    if len(freqs) == 0:
        return np.nan

    # cannot resolve a period longer than half the array
    max_period = n_copies / 2

    peaks, _ = sig.find_peaks(spectrum, height=height_frac * spectrum.max())
    if len(peaks) == 0:
        return np.nan

    periods = 1.0 / freqs[peaks]
    valid = (periods >= min_period) & (periods <= max_period)
    if not valid.any():
        return np.nan

    return float(periods[valid].max())


# ─── Part 4: Pairs file reader ────────────────────────────────────────────────

def read_pairs_file(pairs_file, chrom, array_start, array_end,
                    both_ends=True, chunksize=1_000_000):
    """
    Read an Omni-C or Hi-C .pairs file and pull out contacts at the array.

    A .pairs file has one contact per line:
      readID  chr1  pos1  chr2  pos2  strand1  strand2

    Real files run to tens of millions of lines, so this reads in chunks rather
    than loading everything into memory at once.

    both_ends defaults to True, meaning only contacts where both ends fall
    inside the array are kept. That is what the contact matrix needs, since a
    contact with one end elsewhere has no copy index to assign at that end.
    Set it False to also capture contacts between the array and the rest of the
    genome, which is a different question.
    """
    cols = ['readID', 'chr1', 'pos1', 'chr2', 'pos2', 'strand1', 'strand2']
    kept = []

    reader = pd.read_csv(pairs_file, sep=r'\s+', comment='#', header=None,
                         names=cols, usecols=['chr1', 'pos1', 'chr2', 'pos2'],
                         chunksize=chunksize, dtype={'chr1': str, 'chr2': str})

    for chunk in reader:
        in1 = ((chunk['chr1'] == chrom)
               & chunk['pos1'].between(array_start, array_end))
        in2 = ((chunk['chr2'] == chrom)
               & chunk['pos2'].between(array_start, array_end))

        mask = (in1 & in2) if both_ends else (in1 | in2)
        if mask.any():
            kept.append(chunk[mask])

    if not kept:
        return pd.DataFrame(columns=['chr1', 'pos1', 'chr2', 'pos2'])

    return pd.concat(kept, ignore_index=True)


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def analyse_array(contacts_df, array_start, period, n_copies, label=""):
    """
    Run the whole pipeline and return everything needed to plot or report.
    """
    matrix = build_contact_matrix(contacts_df, array_start, period, n_copies)
    freqs, spectrum, signal_used = fourier_analysis(matrix)
    period_found = dominant_period(freqs, spectrum, n_copies)

    n_contacts = int(matrix.sum() - np.trace(matrix)) // 2 + int(np.trace(matrix))

    return {
        'label': label,
        'matrix': matrix,
        'frequencies': freqs,
        'spectrum': spectrum,
        'signal': signal_used,
        'period': period_found,
        'n_contacts': n_contacts,
    }


def plot_analysis(result, outfile=None):
    """Three-panel summary: matrix, separation profile, spectrum."""
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))

    im = axes[0].imshow(result['matrix'], cmap='Reds', aspect='auto')
    plt.colorbar(im, ax=axes[0], label='Contacts')
    axes[0].set_title('Contact matrix\n(repeat-unit coordinates)')
    axes[0].set_xlabel('Repeat unit')
    axes[0].set_ylabel('Repeat unit')

    axes[1].plot(result['signal'], color='steelblue', lw=2)
    axes[1].axhline(0, color='grey', lw=0.8, ls='--')
    axes[1].set_title('Contact frequency vs separation\n(decay removed)')
    axes[1].set_xlabel('Separation (repeat units)')
    axes[1].set_ylabel('Residual contact frequency')
    axes[1].grid(alpha=0.3)

    if len(result['frequencies']):
        axes[2].plot(result['frequencies'], result['spectrum'],
                     color='coral', lw=2)
        if not np.isnan(result['period']):
            axes[2].axvline(1 / result['period'], color='black', ls='--',
                            lw=1.2, label=f"period {result['period']:.1f}")
            axes[2].legend(fontsize=9)
    axes[2].set_title('Fourier spectrum')
    axes[2].set_xlabel('Frequency (1 / repeat units)')
    axes[2].set_ylabel('Power')
    axes[2].grid(alpha=0.3)

    title = 'RepeatContact'
    if result['label']:
        title += f" — {result['label']}"
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()

    if outfile:
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
    plt.show()


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("RepeatContact v0.2")
    print("=" * 45)

    # MUC1 array in CHM13, from the TRF analysis
    # array spans 63108 to 65430, which is 2322bp, so 38 complete 60bp copies
    ARRAY_START = 63108
    ARRAY_END = 65430
    PERIOD = 60
    N_COPIES = 38

    print(f"\nArray: {ARRAY_START} to {ARRAY_END}")
    print(f"Period: {PERIOD}bp")
    print(f"Complete copies: {N_COPIES}")

    print("\nCoordinate transformation:")
    for pos in [63108, 63168, 63198, 64308, 65430]:
        unit = genomic_to_repeat_unit(pos, ARRAY_START, PERIOD)
        print(f"  {pos} -> unit {unit:.2f}")

    print("\nContact matrix, small worked example:")
    example = pd.DataFrame({
        'pos1': [63108, 63168, 63228, 63528, 63108, 63648],
        'pos2': [63168, 63228, 63288, 63588, 63528, 63708],
    })
    m = build_contact_matrix(example, ARRAY_START, PERIOD, N_COPIES)
    print(f"  shape {m.shape}, {int(m.sum() / 2)} contacts placed, "
          f"{np.count_nonzero(m)} non-zero cells")

    print("\nNote: periodicity detection needs roughly 1000 or more contacts")
    print("inside the array to be reliable. See sensitivity_test.py for the")
    print("full characterisation. This example is far too sparse to interpret.")