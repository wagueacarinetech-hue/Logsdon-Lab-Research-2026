#!/usr/bin/env python3
"""
RepeatContact: A tool for analyzing 3D chromatin contacts
within tandem repeat arrays using repeat-unit coordinates.

Author: Carine Waguea Fongang
Logsdon Lab, University of Pennsylvania
Summer 2026
"""

import pandas as pd
import numpy as np
from scipy import fft
import matplotlib.pyplot as plt

# ─── Part 1: Coordinate Transformer ──────────────────────────────────────────

def genomic_to_repeat_unit(position, array_start, period):
    """
    Convert a genomic position to a repeat unit index.
    
    array_start: where the repeat array begins
    period: size of one repeat unit in bp (60 for MUC1)
    
    Example:
    array starts at 63108, period = 60bp
    position 63168 → repeat unit 1.0
    position 63228 → repeat unit 2.0
    """
    offset = position - array_start
    if offset < 0:
        return None
    return offset / period

# ─── Part 2: Contact Matrix Builder ──────────────────────────────────────────

def build_contact_matrix(contacts_df, array_start, period, n_copies):
    """
    Build a contact matrix in repeat-unit coordinates.
    
    contacts_df: dataframe with columns pos1 and pos2
    array_start: where the repeat array begins
    period: size of one repeat unit in bp
    n_copies: total number of copies in the array
    
    Returns: numpy matrix where matrix[i][j] =
             number of contacts between copy i and copy j
    """
    matrix = np.zeros((n_copies, n_copies))
    
    for _, row in contacts_df.iterrows():
        unit1 = genomic_to_repeat_unit(row['pos1'], array_start, period)
        unit2 = genomic_to_repeat_unit(row['pos2'], array_start, period)
        
        if unit1 is None or unit2 is None:
            continue
        if unit1 >= n_copies or unit2 >= n_copies:
            continue
        
        i = int(unit1)
        j = int(unit2)
        
        matrix[i][j] += 1
        matrix[j][i] += 1
    
    return matrix

# ─── Part 3: Fourier Analysis ─────────────────────────────────────────────────

def fourier_analysis(matrix):
    """
    Apply Fourier transform to the contact matrix
    to find periodic folding patterns.
    
    A peak at frequency 1/N means every Nth copy
    contacts every other Nth copy — revealing
    higher order organization of the repeat array.
    """
    n = matrix.shape[0]
    diagonal_sums = []
    
    for offset in range(n):
        diagonal = np.diagonal(matrix, offset=offset)
        diagonal_sums.append(np.mean(diagonal))
    
    signal = np.array(diagonal_sums)
    
    # apply FFT
    spectrum = np.abs(fft.fft(signal))
    frequencies = fft.fftfreq(len(signal))
    
    # keep only positive frequencies
    pos_mask = frequencies > 0
    frequencies = frequencies[pos_mask]
    spectrum = spectrum[pos_mask]
    
    return frequencies, spectrum, signal

# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("RepeatContact v0.1")
    print("=" * 40)
    
    # MUC1 array in CHM13 from our TRF results
    array_start = 63108
    period = 60
    n_copies = 39
    
    print(f"\nArray start: {array_start}")
    print(f"Period: {period}bp")
    
    # Part 1 test
    print("\nCoordinate transformation test:")
    test_positions = [63108, 63168, 63228, 63528, 64308, 65430]
    for pos in test_positions:
        unit = genomic_to_repeat_unit(pos, array_start, period)
        print(f"  Genomic position {pos} → repeat unit {unit:.1f}")
    
    # Part 2 test
    print("\nContact matrix test:")
    fake_contacts = pd.DataFrame({
        'pos1': [63108, 63168, 63228, 63528, 63108, 63648],
        'pos2': [63168, 63228, 63288, 63588, 63528, 63708]
    })
    
    matrix = build_contact_matrix(fake_contacts, array_start, period, n_copies)
    
    print(f"Contact matrix shape: {matrix.shape}")
    print(f"Total contacts: {int(matrix.sum() / 2)}")
    print(f"Non-zero entries: {np.count_nonzero(matrix)}")
    
    # Part 3 test
    print("\nFourier analysis test:")
    frequencies, spectrum, signal = fourier_analysis(matrix)
    print(f"Number of frequencies analyzed: {len(frequencies)}")
    print(f"Dominant frequency: {frequencies[np.argmax(spectrum)]:.3f}")
    print(f"Dominant period: {1/frequencies[np.argmax(spectrum)]:.1f} repeat units")
    
    # plot all three panels
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # panel 1 - contact matrix heatmap
    axes[0].imshow(matrix, cmap='Reds', aspect='auto')
    axes[0].set_title('Contact Matrix\n(repeat-unit coordinates)')
    axes[0].set_xlabel('Repeat unit index')
    axes[0].set_ylabel('Repeat unit index')
    
    # panel 2 - diagonal signal
    axes[1].plot(signal, color='steelblue')
    axes[1].set_title('Contact frequency\nvs separation distance')
    axes[1].set_xlabel('Separation (repeat units)')
    axes[1].set_ylabel('Mean contact frequency')
    
    # panel 3 - Fourier spectrum
    axes[2].plot(frequencies, spectrum, color='coral')
    axes[2].set_title('Fourier spectrum\n(periodic folding patterns)')
    axes[2].set_xlabel('Frequency (1/repeat units)')
    axes[2].set_ylabel('Power')
    
    plt.suptitle('RepeatContact — MUC1 VNTR Analysis (test data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/wagueacarinefongang/Documents/RepeatContact/repeatcontact_output.png', dpi=150)
    plt.show()
    print("\nFull analysis plot saved!")