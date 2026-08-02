#!/usr/bin/env python3
"""
Can RepeatContact detect a folding period, and how much data does it need?

This generates synthetic contacts where I control the answer, then checks
whether the pipeline recovers it. Two things are being tested:

  1. Does the method find a periodicity that is genuinely there?
  2. Does it stay quiet when there is no periodicity?

A method that fails the second test is worse than useless, because it would
report structure in random data.

The sweep over contact count answers the practical question: how many real
contacts would I need at a locus this size for the answer to mean anything.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import fft, signal as sig

from repeatcontact import build_contact_matrix, fourier_analysis

# ─── Configuration ────────────────────────────────────────────────────────────

ARRAY_START = 63108
PERIOD_BP = 60
N_COPIES = 38          # 2322bp / 60 = 38.7, so 38 complete units

TRUE_FOLD_PERIOD = 5   # the answer the method should recover

CONTACT_COUNTS = [100, 500, 1000, 5000, 20000]
SIGNAL_STRENGTHS = [0.0, 0.10, 0.25, 0.50]

N_REPLICATES = 20      # repeat each condition to see how often it works

OUTPUT = "/Users/wagueacarinefongang/Documents/RepeatContact/sensitivity_test.png"


# ─── Simulating contacts ──────────────────────────────────────────────────────

def simulate_contacts(n_contacts, fold_period=None, strength=0.0,
                      decay=1.0, rng=None):
    """
    Build a synthetic set of contacts inside the array.

    Two components:

    Background. Contact probability falls off with separation, which is what
    real chromatin does. P(contact between copies i and j) is proportional to
    1 / |i-j|^decay. Every contact set has this.

    Periodic. A fraction of contacts, set by `strength`, are placed at
    separations that are exact multiples of `fold_period`. This is the signal
    the method is supposed to find. With strength = 0 there is no signal, and
    the method should report nothing.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_periodic = int(n_contacts * strength)
    n_background = n_contacts - n_periodic

    pairs = []

    # background: separation drawn with 1/d decay
    seps = np.arange(1, N_COPIES)
    weights = 1.0 / seps**decay
    weights /= weights.sum()

    for _ in range(n_background):
        d = rng.choice(seps, p=weights)
        i = rng.integers(0, N_COPIES - d)
        pairs.append((i, i + d))

    # periodic: separation is a multiple of fold_period
    if n_periodic > 0 and fold_period:
        multiples = np.arange(fold_period, N_COPIES, fold_period)
        for _ in range(n_periodic):
            d = rng.choice(multiples)
            i = rng.integers(0, N_COPIES - d)
            pairs.append((i, i + d))

    # convert copy indices back to genomic positions, with a random offset
    # inside each unit, because that is what real data would look like
    rows = []
    for i, j in pairs:
        p1 = ARRAY_START + i * PERIOD_BP + rng.integers(0, PERIOD_BP)
        p2 = ARRAY_START + j * PERIOD_BP + rng.integers(0, PERIOD_BP)
        rows.append({'pos1': p1, 'pos2': p2})

    return pd.DataFrame(rows)


# ─── Detrended Fourier analysis ───────────────────────────────────────────────

def fourier_detrended(matrix):
    """
    Same idea as fourier_analysis in the main tool, but removing the decay
    trend before transforming.

    The reason this matters: contact frequency falls off steeply with
    separation, and that falloff is a large low-frequency component. Without
    removing it, the FFT reports the decay curve rather than any periodicity
    sitting on top of it, so the dominant frequency comes out near the array
    length regardless of the input.

    Subtracting a fitted trend leaves the residual, which is where a genuine
    periodic signal lives.
    """
    n = matrix.shape[0]
    means = np.array([np.mean(np.diagonal(matrix, offset=k)) for k in range(n)])

    # fit and subtract the decay. working in log space because the falloff is
    # closer to a power law than a straight line
    x = np.arange(1, n)
    y = means[1:]
    ok = y > 0
    if ok.sum() < 5:
        return np.array([]), np.array([]), means

    coef = np.polyfit(np.log(x[ok]), np.log(y[ok]), 1)
    trend = np.exp(np.polyval(coef, np.log(x)))
    residual = y - trend

    # taper the ends so the FFT does not see an artificial step
    windowed = residual * np.hanning(len(residual))

    spectrum = np.abs(fft.fft(windowed))
    freqs = fft.fftfreq(len(windowed))

    keep = freqs > 0
    return freqs[keep], spectrum[keep], residual


def recovered_period(freqs, spectrum, min_period=2.5, max_period=None):
    """
    Pick the fundamental frequency, not a harmonic.

    A signal with period 5 also produces power at 2.5, 1.67 and so on, because
    a repeating pattern is not a pure sine wave. Those harmonics can be taller
    than the fundamental, so taking the global maximum reports the wrong
    answer, and reports it more reliably as the data improves.

    Instead: find every local peak above a fraction of the maximum, then take
    the one with the longest period. Harmonics always sit at shorter periods
    than the fundamental they come from.
    """
    if len(freqs) == 0:
        return np.nan

    if max_period is None:
        max_period = N_COPIES / 2   # cannot resolve a period longer than this

    peaks, props = sig.find_peaks(spectrum, height=0.2 * spectrum.max())
    if len(peaks) == 0:
        return np.nan

    periods = 1.0 / freqs[peaks]
    valid = (periods >= min_period) & (periods <= max_period)
    if not valid.any():
        return np.nan

    return periods[valid].max()


# ─── Sweep ────────────────────────────────────────────────────────────────────

def run_sweep():
    """
    For every combination of contact count and signal strength, simulate the
    data many times and count how often the true folding period is recovered.

    Counting it as recovered if the answer lands within 15% of the truth,
    since the frequency resolution is limited by array length.
    """
    rng = np.random.default_rng(42)

    results_plain = np.zeros((len(SIGNAL_STRENGTHS), len(CONTACT_COUNTS)))
    results_detrend = np.zeros_like(results_plain)

    for si, strength in enumerate(SIGNAL_STRENGTHS):
        for ci, n_contacts in enumerate(CONTACT_COUNTS):
            hits_plain = 0
            hits_detrend = 0

            for _ in range(N_REPLICATES):
                df = simulate_contacts(n_contacts, TRUE_FOLD_PERIOD,
                                       strength, rng=rng)
                m = build_contact_matrix(df, ARRAY_START, PERIOD_BP, N_COPIES)

                f1, s1, _ = fourier_analysis(m, detrend=False)
                p1 = recovered_period(f1, s1)
                if abs(p1 - TRUE_FOLD_PERIOD) / TRUE_FOLD_PERIOD < 0.15:
                    hits_plain += 1

                f2, s2, _ = fourier_detrended(m)
                p2 = recovered_period(f2, s2)
                if abs(p2 - TRUE_FOLD_PERIOD) / TRUE_FOLD_PERIOD < 0.15:
                    hits_detrend += 1

            results_plain[si, ci] = hits_plain / N_REPLICATES
            results_detrend[si, ci] = hits_detrend / N_REPLICATES

            print(f"  strength={strength:<5} contacts={n_contacts:<6} "
                  f"plain={results_plain[si,ci]:.2f}  "
                  f"detrended={results_detrend[si,ci]:.2f}")

    return results_plain, results_detrend


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    print("RepeatContact sensitivity test")
    print("=" * 55)
    print(f"Array: {N_COPIES} copies of {PERIOD_BP}bp")
    print(f"True folding period built into the data: every {TRUE_FOLD_PERIOD} copies")
    print(f"Replicates per condition: {N_REPLICATES}")
    print()

    # first, a single clear demonstration at high signal and high depth
    rng = np.random.default_rng(1)
    df = simulate_contacts(20000, TRUE_FOLD_PERIOD, 0.5, rng=rng)
    m = build_contact_matrix(df, ARRAY_START, PERIOD_BP, N_COPIES)

    f_plain, s_plain, sig_plain = fourier_analysis(m)
    f_det, s_det, sig_det = fourier_detrended(m)

    print("Best case, 20000 contacts with half of them periodic:")
    print(f"  true period               : {TRUE_FOLD_PERIOD}")
    print(f"  recovered, current method : {recovered_period(f_plain, s_plain):.1f}")
    print(f"  recovered, detrended      : {recovered_period(f_det, s_det):.1f}")
    print()

    # null check: no periodicity at all, method should not find one
    df_null = simulate_contacts(20000, None, 0.0, rng=rng)
    m_null = build_contact_matrix(df_null, ARRAY_START, PERIOD_BP, N_COPIES)
    f_null, s_null, _ = fourier_detrended(m_null)
    print(f"Null data with no periodicity, detrended method reports period "
          f"{recovered_period(f_null, s_null):.1f}")
    print("  (should not be close to 5, otherwise the method invents structure)")
    print()

    print("Sweep, fraction of runs recovering the true period:")
    plain, detrend = run_sweep()

    # ─── Figure ───────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # contact matrix
    im = axes[0, 0].imshow(m, cmap='Reds', aspect='auto')
    plt.colorbar(im, ax=axes[0, 0], label='Contacts')
    axes[0, 0].set_title(f"Contact matrix, periodicity of {TRUE_FOLD_PERIOD} "
                         "built in", fontweight='bold')
    axes[0, 0].set_xlabel("Repeat unit")
    axes[0, 0].set_ylabel("Repeat unit")

    # the signal before and after detrending
    axes[0, 1].plot(sig_plain, color='steelblue', lw=2, label='raw diagonal means')
    axes[0, 1].plot(range(1, len(sig_det) + 1), sig_det, color='coral', lw=2,
                    label='after removing decay')
    axes[0, 1].axhline(0, color='grey', lw=0.8, ls='--')
    axes[0, 1].set_title("Why detrending is necessary", fontweight='bold')
    axes[0, 1].set_xlabel("Separation (repeat units)")
    axes[0, 1].set_ylabel("Mean contact frequency")
    axes[0, 1].legend(fontsize=9)
    axes[0, 1].grid(alpha=0.3)

    # spectra
    axes[1, 0].plot(f_plain, s_plain / s_plain.max(), color='steelblue',
                    lw=2, label='current method')
    axes[1, 0].plot(f_det, s_det / s_det.max(), color='coral',
                    lw=2, label='detrended')
    axes[1, 0].axvline(1 / TRUE_FOLD_PERIOD, color='black', ls='--', lw=1.2,
                       label=f'true period ({TRUE_FOLD_PERIOD})')
    axes[1, 0].set_title("Fourier spectrum", fontweight='bold')
    axes[1, 0].set_xlabel("Frequency (1 / repeat units)")
    axes[1, 0].set_ylabel("Normalised power")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.3)

    # detection rate grid
    im2 = axes[1, 1].imshow(detrend, cmap='YlGn', vmin=0, vmax=1, aspect='auto')
    axes[1, 1].set_xticks(range(len(CONTACT_COUNTS)))
    axes[1, 1].set_xticklabels(CONTACT_COUNTS)
    axes[1, 1].set_yticks(range(len(SIGNAL_STRENGTHS)))
    axes[1, 1].set_yticklabels([f"{s:.0%}" for s in SIGNAL_STRENGTHS])
    axes[1, 1].set_xlabel("Number of contacts")
    axes[1, 1].set_ylabel("Fraction of contacts that are periodic")
    axes[1, 1].set_title("Detection rate, detrended method", fontweight='bold')
    for i in range(len(SIGNAL_STRENGTHS)):
        for j in range(len(CONTACT_COUNTS)):
            axes[1, 1].text(j, i, f"{detrend[i,j]:.2f}", ha='center',
                            va='center', fontsize=9,
                            color='white' if detrend[i, j] > 0.6 else 'black')
    plt.colorbar(im2, ax=axes[1, 1], label='Fraction of runs recovering period')

    plt.suptitle("RepeatContact: can it detect a folding period, and with how much data?",
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"\nSaved: {OUTPUT}")