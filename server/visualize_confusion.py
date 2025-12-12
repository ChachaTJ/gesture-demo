#!/usr/bin/env python3
"""
Phoneme Confusion Matrix Visualizer
Generates a heatmap from the model's confusion matrix data.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Phoneme list (matching LOGIT_TO_PHONEME order)
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH'
]

def load_confusion_data(json_path):
    """Load confusion matrix from JSON."""
    with open(json_path, 'r') as f:
        return json.load(f)

def build_matrix(confusion_data):
    """Build NxN confusion matrix from JSON data."""
    cm = confusion_data.get('confusion_matrix', {})
    n = len(PHONEMES)
    matrix = np.zeros((n, n))
    
    for i, gt_phoneme in enumerate(PHONEMES):
        if gt_phoneme in cm:
            row_total = sum(cm[gt_phoneme].values())
            for j, pred_phoneme in enumerate(PHONEMES):
                count = cm[gt_phoneme].get(pred_phoneme, 0)
                # Normalize to percentage
                if row_total > 0:
                    matrix[i, j] = (count / row_total) * 100
    
    return matrix

def calculate_accuracy(confusion_data):
    """Calculate per-phoneme accuracy."""
    cm = confusion_data.get('confusion_matrix', {})
    totals = confusion_data.get('total_samples_per_phoneme', {})
    
    accuracies = {}
    for phoneme in PHONEMES:
        if phoneme in cm and phoneme in totals:
            correct = cm[phoneme].get(phoneme, 0)
            total = totals.get(phoneme, 1)
            accuracies[phoneme] = (correct / total) * 100 if total > 0 else 0
    
    return accuracies

def plot_confusion_matrix(matrix, save_path):
    """Generate heatmap: White→Red (0-10%), White (10-90%), White→Blue (90-100%)."""
    import matplotlib.colors as mcolors
    from matplotlib.colors import LinearSegmentedColormap
    
    plt.figure(figsize=(18, 16))
    
    # Custom colormap: 
    # 0=White -> 10=Red -> 50=White -> 90=White -> 100=Blue
    colors = [
        (0.00, '#ffffff'),  # 0% - White (no confusion)
        (0.10, '#e74c3c'),  # 10% - Red (high confusion)
        (0.50, '#f8f9fa'),  # 50% - Light gray/white
        (0.90, '#ffffff'),  # 90% - White 
        (1.00, '#2980b9'),  # 100% - Blue (perfect)
    ]
    cmap = LinearSegmentedColormap.from_list('custom', 
        [(pos, color) for pos, color in colors])
    
    ax = sns.heatmap(
        matrix,
        xticklabels=PHONEMES,
        yticklabels=PHONEMES,
        cmap=cmap,
        vmin=0, vmax=100,  # No normalization, linear scale
        annot=True,
        fmt='.0f',
        annot_kws={'size': 7, 'color': 'black'},
        square=True,
        linewidths=0.3,
        linecolor='lightgray',
        cbar_kws={'label': 'Accuracy (%)', 'shrink': 0.8}
    )
    
    # Highlight diagonal with dark border
    for i in range(len(PHONEMES)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False, edgecolor='#2c3e50', lw=1.5))
    
    plt.xlabel('Predicted Phoneme', fontsize=12, fontweight='bold')
    plt.ylabel('Ground Truth Phoneme', fontsize=12, fontweight='bold')
    plt.title('Phoneme Confusion Matrix\nWhite→Red (Confusion) | White→Blue (Accuracy)', 
              fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved confusion matrix to: {save_path}")

def plot_accuracy_bar(accuracies, save_path):
    """Generate per-phoneme accuracy bar chart."""
    plt.figure(figsize=(14, 6))
    
    # Sort by accuracy
    sorted_items = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    phonemes = [x[0] for x in sorted_items]
    accs = [x[1] for x in sorted_items]
    
    # Color bars based on accuracy
    colors = ['#7bed9f' if a >= 95 else '#ffa502' if a >= 90 else '#ff4757' for a in accs]
    
    bars = plt.bar(phonemes, accs, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add average line
    avg = np.mean(accs)
    plt.axhline(y=avg, color='#1e3c72', linestyle='--', linewidth=2, label=f'Average: {avg:.1f}%')
    
    plt.xlabel('Phoneme', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    plt.title('Per-Phoneme Accuracy\n(Higher = Better)', fontsize=14, fontweight='bold', pad=15)
    plt.ylim(0, 105)
    plt.legend(loc='lower right')
    
    # Add value labels on top of bars
    for bar, acc in zip(bars, accs):
        if acc < 100:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{acc:.0f}', ha='center', va='bottom', fontsize=7)
    
    plt.xticks(rotation=45, ha='right', fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved accuracy chart to: {save_path}")

def main():
    # Paths
    json_path = Path('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_matrix.json')
    output_dir = Path('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs')
    
    # Load data
    print("Loading confusion matrix data...")
    data = load_confusion_data(json_path)
    
    # Build matrix
    print("Building confusion matrix...")
    matrix = build_matrix(data)
    
    # Calculate accuracies
    accuracies = calculate_accuracy(data)
    
    # Print summary
    print("\n" + "="*50)
    print("📊 PHONEME MODEL ACCURACY REPORT")
    print("="*50)
    
    # Top 5 most accurate
    sorted_acc = sorted(accuracies.items(), key=lambda x: x[1], reverse=True)
    print("\n🏆 Top 5 Most Accurate Phonemes:")
    for p, a in sorted_acc[:5]:
        print(f"   {p}: {a:.1f}%")
    
    # Bottom 5
    print("\n⚠️ 5 Phonemes Needing Improvement:")
    for p, a in sorted_acc[-5:]:
        print(f"   {p}: {a:.1f}%")
    
    # Overall stats
    all_accs = list(accuracies.values())
    print(f"\n📈 Overall Statistics:")
    print(f"   Average Accuracy: {np.mean(all_accs):.1f}%")
    print(f"   Min: {np.min(all_accs):.1f}%")
    print(f"   Max: {np.max(all_accs):.1f}%")
    print(f"   Phonemes ≥95%: {sum(1 for a in all_accs if a >= 95)}/{len(all_accs)}")
    
    # Generate visualizations
    print("\n🎨 Generating visualizations...")
    plot_confusion_matrix(matrix, output_dir / 'confusion_matrix_heatmap.png')
    plot_accuracy_bar(accuracies, output_dir / 'phoneme_accuracy_chart.png')
    
    print("\n✨ Done!")

if __name__ == '__main__':
    main()
