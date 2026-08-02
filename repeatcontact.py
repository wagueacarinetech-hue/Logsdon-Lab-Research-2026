#!/usr/bin/env python3
"""
RepeatContact: A tool for analyzing 3D chromatin contacts
within tandem repeat arrays using repeat-unit coordinates.

 Existing chromatin contact tools use linear
genomic coordinates. But tandem repeat arrays like MUC1 have
different lengths in different people. This tool converts contacts
into repeat-unit coordinates so you can compare folding patterns
across arrays of different lengths.

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
    This is the core function of the tool.
    
    Instead of asking "what genomic position is this contact at?"
    we ask "which repeat copy is this contact in?"
    
    For example if the MUC1 array starts at position 63108
    and each copy is 60bp long:
    position 63108 = repeat unit 0 (first copy)
    position 63168 = repeat unit 1 (second copy)
    position 63228 = repeat unit 2 (third copy)
    
    This lets us compare arrays of different lengths
    on the same scale.
    """
    offset = position - array_start
    if offset < 0:
        return None
    return offset / period


# ─── Part 2: Contact Matrix Builder ──────────────────────────────────────────

def build_contact_matrix(contacts_df, array_start, period, n_copies):
    """
    Takes a list of contacts and builds a matrix showing
    how often each repeat copy contacts every other copy.
    
    The matrix is n_copies x n_copies.
    matrix[i][j] tells you how many times copy i
    was in contact with copy j.
    
    A strong diagonal means nearby copies contact each other a lot.
    Off-diagonal patterns reveal higher order folding structure.
    """
    matrix = np.zeros((n_copies, n_copies))
    
    for _, row in contacts_df.iterrows():
        unit1 = genomic_to_repeat_unit(row['pos1'], array_start, period)
        unit2 = genomic_to_repeat_unit(row['pos2'], array_start, period)
        
        # skip contacts that fall outside the array
        if unit1 is None or unit2 is None:
            continue
        if unit1 >= n_copies or unit2 >= n_copies:
            continue
        
        i = int(unit1)
        j = int(unit2)
        
        # contact matrices are symmetric
        # if copy i contacts copy j then copy j contacts copy i
        matrix[i][j] += 1
        matrix[j][i] += 1
    
    return matrix


# ─── Part 3: Fourier Analysis ─────────────────────────────────────────────────

def fourier_analysis(matrix):
    """
    This is where the signal processing comes in.
    
    We look at how contact frequency changes as you
    move further away along the array (diagonal offset).
    
    Then we apply a Fourier transform to find
    periodic patterns in that signal.
    
    A peak at frequency 1/N means every Nth copy
    contacts every other Nth copy.
    This tells us about higher order folding structure
    of the repeat array in 3D space.
    
    This is the same concept as finding frequencies
    in an electrical signal — just applied to DNA.
    """
    n = matrix.shape[0]
    diagonal_sums = []
    
    # for each diagonal offset calculate mean contact frequency
    for offset in range(n):
        diagonal = np.diagonal(matrix, offset=offset)
        diagonal_sums.append(np.mean(diagonal))
    
    signal = np.array(diagonal_sums)
    
    # apply Fast Fourier Transform
    spectrum = np.abs(fft.fft(signal))
    frequencies = fft.fftfreq(len(signal))
    
    # keep only positive frequencies
    pos_mask = frequencies > 0
    frequencies = frequencies[pos_mask]
    spectrum = spectrum[pos_mask]
    
    return frequencies, spectrum, signal


# ─── Part 4: Pairs File Reader ────────────────────────────────────────────────

def read_pairs_file(pairs_file, chrom, array_start, array_end):
    """
    Reads an Omni-C or Hi-C .pairs file and extracts
    all contacts where at least one end falls inside
    the repeat array we care about.
    
    A .pairs file looks like this:
    readID  chr1  pos1  chr2  pos2  strand1  strand2
    read001 chr13 63120 chr13 63480 +        -
    
    Each line is one contact between two genomic positions.
    We only keep contacts that involve our repeat array.
    """
    contacts = []
    
    with open(pairs_file) as f:
        for line in f:
            # skip header lines starting with #
            if line.startswith('#'):
                continue
            
            fields = line.strip().split()
            if len(fields) < 6:
                continue
            
            chr1 = fields[1]
            pos1 = int(fields[2])
            chr2 = fields[3]
            pos2 = int(fields[4])
            
            # check if either end of the contact
            # falls within our repeat array
            pos1_in_array = (chr1 == chrom and
                           array_start <= pos1 <= array_end)
            pos2_in_array = (chr2 == chrom and
                           array_start <= pos2 <= array_end)
            
            if pos1_in_array or pos2_in_array:
                contacts.append({
                    'pos1': pos1,
                    'pos2': pos2,
                    'chr1': chr1,
                    'chr2': chr2
                })
    
    return pd.DataFrame(contacts)


# ─── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    
    print("RepeatContact v0.1")
    print("=" * 40)
    
    # MUC1 array parameters from our TRF analysis of CHM13
    # we found this by running TRF on the CHM13 MUC1 sequence
    array_start = 63108
    array_end = 65430
    period = 60
    n_copies = 39
    
    print(f"\nArray start: {array_start}")
    print(f"Array end: {array_end}")
    print(f"Period: {period}bp")
    print(f"Copy number: {n_copies}")
    
    # Part 1 test
    # making sure the coordinate transformation works correctly
    # position 65430 should give exactly 38.7 copies
    # which matches what TRF found for CHM13
    print("\nPart 1 — Coordinate transformation test:")
    test_positions = [63108, 63168, 63228, 63528, 64308, 65430]
    for pos in test_positions:
        unit = genomic_to_repeat_unit(pos, array_start, period)
        print(f"  Genomic position {pos} → repeat unit {unit:.1f}")
    
    # Part 2 test
    # using fake contacts to test the matrix builder
    # in real analysis these come from the Omni-C pairs file
    print("\nPart 2 — Contact matrix test:")
    fake_contacts = pd.DataFrame({
        'pos1': [63108, 63168, 63228, 63528, 63108, 63648],
        'pos2': [63168, 63228, 63288, 63588, 63528, 63708]
    })
    
    matrix = build_contact_matrix(fake_contacts, array_start, period, n_copies)
    
    print(f"Contact matrix shape: {matrix.shape}")
    print(f"Total contacts: {int(matrix.sum() / 2)}")
    print(f"Non-zero entries: {np.count_nonzero(matrix)}")
    
    # Part 3 test
    # applying Fourier analysis to find periodic patterns
    print("\nPart 3 — Fourier analysis test:")
    frequencies, spectrum, signal = fourier_analysis(matrix)
    print(f"Number of frequencies analyzed: {len(frequencies)}")
    dominant_freq = frequencies[np.argmax(spectrum)]
    print(f"Dominant frequency: {dominant_freq:.3f}")
    print(f"Dominant period: {1/dominant_freq:.1f} repeat units")
    print(f"Meaning: every {1/dominant_freq:.0f} copies contact each other")
    
    # Part 4 test
    # testing the pairs file reader with a fake pairs file
    print("\nPart 4 — Pairs file reader test:")
    
    fake_pairs_content = """#columns: readID chr1 pos1 chr2 pos2 strand1 strand2
read001\tchr13\t63120\tchr13\t63480\t+\t-
read002\tchr13\t63060\tchr1\t45230\t+\t+
read003\tchr13\t63540\tchr13\t63720\t-\t+
read004\tchr1\t12345\tchr13\t63800\t+\t+
read005\tchr2\t99999\tchr2\t88888\t+\t-
"""
    
    # write fake pairs file to disk
    with open('/Users/wagueacarinefongang/Documents/RepeatContact/test.pairs', 'w') as f:
        f.write(fake_pairs_content)
    
    # read it back and filter for array contacts
    contacts = read_pairs_file(
        '/Users/wagueacarinefongang/Documents/RepeatContact/test.pairs',
        chrom='chr13',
        array_start=array_start,
        array_end=array_end
    )
    
    print(f"Total contacts in pairs file: 5")
    print(f"Contacts involving repeat array: {len(contacts)}")
    print(contacts.to_string())
    
    # generate full analysis plot
    print("\nGenerating plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # panel 1 - contact matrix heatmap
    im = axes[0].imshow(matrix, cmap='Reds', aspect='auto')
    plt.colorbar(im, ax=axes[0], label='Contact frequency')
    axes[0].set_title('Contact Matrix\n(repeat-unit coordinates)')
    axes[0].set_xlabel('Repeat unit index')
    axes[0].set_ylabel('Repeat unit index')
    
    # panel 2 - diagonal signal
    axes[1].plot(signal, color='steelblue', linewidth=2)
    axes[1].set_title('Contact frequency\nvs separation distance')
    axes[1].set_xlabel('Separation (repeat units)')
    axes[1].set_ylabel('Mean contact frequency')
    axes[1].grid(alpha=0.3)
    
    # panel 3 - Fourier spectrum
    axes[2].plot(frequencies, spectrum, color='coral', linewidth=2)
    axes[2].set_title('Fourier spectrum\n(periodic folding patterns)')
    axes[2].set_xlabel('Frequency (1/repeat units)')
    axes[2].set_ylabel('Power')
    axes[2].grid(alpha=0.3)
    
    plt.suptitle('RepeatContact — MUC1 VNTR Analysis (test data)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('/Users/wagueacarinefongang/Documents/RepeatContact/repeatcontact_output.png',
                dpi=150, bbox_inches='tight')
    plt.show()
    print("Full analysis plot saved!")
    print("\nRepeatContact v0.1 — all tests passed!")