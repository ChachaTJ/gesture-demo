#!/usr/bin/env python3
"""
Create visualization of LibriSpeech Phoneme Frequencies
"""

import matplotlib.pyplot as plt
import numpy as np

# Data from 5M line analysis (365M phonemes)
phonemes = ['ZH', 'OY', 'UH', 'JH', 'TH', 'CH', 'Y', 'AW', 'SH', 'G', 
            'NG', 'OW', 'EY', 'AO', 'AY', 'AA', 'P', 'UW', 'F', 'B',
            'V', 'HH', 'W', 'ER', 'K', 'Z', 'EH', 'M', 'AE', 'IY',
            'DH', 'L', 'R', 'S', 'D', 'IH', 'T', 'N', 'AH']

frequencies = [0.044, 0.110, 0.449, 0.450, 0.468, 0.563, 0.622, 0.649, 0.768, 0.867,
               1.114, 1.261, 1.528, 1.664, 1.690, 1.789, 1.819, 1.853, 1.937, 1.952,
               2.005, 2.165, 2.214, 2.606, 2.632, 2.801, 2.803, 2.907, 2.992, 3.193,
               3.250, 3.975, 4.312, 4.592, 5.190, 5.843, 7.031, 7.270, 10.622]

# Create figure with 2 subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10))
fig.suptitle('LibriSpeech Phoneme Frequency Analysis\n(5M lines, 365M phonemes)', 
             fontsize=16, fontweight='bold', y=0.98)

# ===== Plot 1: Full ranking bar chart =====
colors = []
for f in frequencies:
    if f < 0.5:
        colors.append('#27ae60')  # Green (rare - best triggers)
    elif f < 1.0:
        colors.append('#f39c12')  # Yellow (somewhat rare)
    elif f < 3.0:
        colors.append('#3498db')  # Blue (common)
    else:
        colors.append('#e74c3c')  # Red (very common)

ax1.barh(range(len(phonemes)), frequencies, color=colors, edgecolor='white', linewidth=0.5)
ax1.set_yticks(range(len(phonemes)))
ax1.set_yticklabels(phonemes, fontsize=9, fontfamily='monospace')
ax1.set_xlabel('Frequency (%)', fontsize=12, fontweight='bold')
ax1.set_title('All Phonemes (Rare → Common)', fontsize=12, fontweight='bold')
ax1.invert_yaxis()

# Add frequency labels
for i, (p, f) in enumerate(zip(phonemes, frequencies)):
    ax1.text(f + 0.1, i, f'{f:.2f}%', va='center', fontsize=8)

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27ae60', label='Best Triggers (<0.5%)'),
    Patch(facecolor='#f39c12', label='Good Triggers (0.5-1%)'),
    Patch(facecolor='#3498db', label='Common (1-3%)'),
    Patch(facecolor='#e74c3c', label='Very Common (>3%)')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=9)

# ===== Plot 2: Top 10 Trigger Candidates =====
top_phonemes = phonemes[:10]
top_freqs = frequencies[:10]
top_colors = colors[:10]

bars = ax2.bar(range(len(top_phonemes)), top_freqs, color=top_colors, edgecolor='black', linewidth=1)
ax2.set_xticks(range(len(top_phonemes)))
ax2.set_xticklabels(top_phonemes, fontsize=12, fontweight='bold', fontfamily='monospace')
ax2.set_ylabel('Frequency (%)', fontsize=12, fontweight='bold')
ax2.set_title('🎯 Top 10 Trigger Candidates\n(Lower = Better)', fontsize=12, fontweight='bold')

# Add value labels on bars
for bar, f in zip(bars, top_freqs):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
             f'{f:.3f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Add example pronunciations
examples = ['Vision', 'Boy', 'Book', 'Joy', 'Think', 'Cheese', 'Yes', 'Out', 'She', 'Go']
for i, ex in enumerate(examples):
    ax2.text(i, -0.1, f'({ex})', ha='center', va='top', fontsize=8, color='gray')

ax2.set_ylim(0, max(top_freqs) * 1.2)

# ===== Save =====
plt.tight_layout()
plt.savefig('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/phoneme_frequency_chart.png', 
            dpi=150, bbox_inches='tight', facecolor='white')
plt.close()

print("✓ Saved: outputs/phoneme_frequency_chart.png")
