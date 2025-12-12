#!/usr/bin/env python3
"""
Enhanced Confusion Matrix Visualization
- Highlights phonemes with LOW confusion (good triggers)
- Shows "Distinctiveness Score" (diagonal - off-diagonal)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Phoneme list
PHONEMES = [
    'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH'
]

def load_data():
    with open('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_matrix.json', 'r') as f:
        return json.load(f)

def calculate_distinctiveness(data):
    """Calculate how distinct each phoneme is (low confusion with others)."""
    cm = data.get('confusion_matrix', {})
    totals = data.get('total_samples_per_phoneme', {})
    
    scores = {}
    for phoneme in PHONEMES:
        if phoneme not in cm or phoneme not in totals:
            continue
        
        total = totals[phoneme]
        if total == 0:
            continue
        
        # Correct predictions (diagonal)
        correct = cm[phoneme].get(phoneme, 0)
        accuracy = correct / total
        
        # How often OTHER phonemes get mistaken for this one (false positives)
        false_positives = 0
        total_others = 0
        for other_p, predictions in cm.items():
            if other_p != phoneme:
                false_positives += predictions.get(phoneme, 0)
                total_others += totals.get(other_p, 0)
        
        # Specificity: 1 - (false positives / total others)
        specificity = 1 - (false_positives / total_others) if total_others > 0 else 1
        
        # Distinctiveness = Accuracy + Specificity (both high = good trigger)
        distinctiveness = (accuracy + specificity) / 2
        
        scores[phoneme] = {
            'accuracy': accuracy * 100,
            'specificity': specificity * 100,
            'distinctiveness': distinctiveness * 100,
            'false_positives': false_positives,
            'total': total
        }
    
    return scores

def plot_enhanced(data, scores):
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('Phoneme Confusion Analysis - Trigger Candidates', fontsize=16, fontweight='bold', y=0.98)
    
    # ===== Plot 1: Distinctiveness Score (높을수록 좋은 Trigger) =====
    ax1 = axes[0, 0]
    sorted_scores = sorted(scores.items(), key=lambda x: x[1]['distinctiveness'], reverse=True)
    phonemes = [p for p, _ in sorted_scores]
    distinctiveness = [s['distinctiveness'] for _, s in sorted_scores]
    
    colors = ['#27ae60' if d >= 99 else '#f39c12' if d >= 97 else '#e74c3c' for d in distinctiveness]
    bars = ax1.barh(phonemes[::-1], distinctiveness[::-1], color=colors[::-1], edgecolor='white')
    
    ax1.axvline(x=99, color='green', linestyle='--', alpha=0.7, label='Excellent (≥99%)')
    ax1.axvline(x=97, color='orange', linestyle='--', alpha=0.7, label='Good (≥97%)')
    ax1.set_xlabel('Distinctiveness Score (%)', fontweight='bold')
    ax1.set_title('Distinctiveness Score\n(Higher = Less Confusion = Better Trigger)', fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.set_xlim(90, 100.5)
    
    # ===== Plot 2: Top 10 Best Triggers =====
    ax2 = axes[0, 1]
    top10 = sorted_scores[:10]
    
    x = np.arange(len(top10))
    width = 0.35
    
    acc = [s['accuracy'] for _, s in top10]
    spec = [s['specificity'] for _, s in top10]
    names = [p for p, _ in top10]
    
    bars1 = ax2.bar(x - width/2, acc, width, label='Accuracy (Recall)', color='#3498db')
    bars2 = ax2.bar(x + width/2, spec, width, label='Specificity (Not Confused)', color='#27ae60')
    
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, fontsize=11, fontweight='bold')
    ax2.set_ylabel('Score (%)', fontweight='bold')
    ax2.set_title('Top 10 Trigger Candidates\n(High Accuracy + High Specificity)', fontweight='bold')
    ax2.legend()
    ax2.set_ylim(90, 101)
    
    # Add value labels
    for bar, val in zip(bars1, acc):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}', ha='center', fontsize=8)
    for bar, val in zip(bars2, spec):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, 
                f'{val:.1f}', ha='center', fontsize=8)
    
    # ===== Plot 3: Confusion Heatmap (Top 10 Only) =====
    ax3 = axes[1, 0]
    top_phonemes = [p for p, _ in top10]
    cm = data.get('confusion_matrix', {})
    totals = data.get('total_samples_per_phoneme', {})
    
    # Build mini confusion matrix
    matrix = np.zeros((10, 10))
    for i, gt in enumerate(top_phonemes):
        if gt in cm and gt in totals:
            total = totals[gt]
            for j, pred in enumerate(top_phonemes):
                count = cm[gt].get(pred, 0)
                matrix[i, j] = (count / total) * 100 if total > 0 else 0
    
    sns.heatmap(matrix, ax=ax3, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=top_phonemes, yticklabels=top_phonemes,
                cbar_kws={'label': '%'}, square=True, linewidths=0.5)
    ax3.set_xlabel('Predicted', fontweight='bold')
    ax3.set_ylabel('Ground Truth', fontweight='bold')
    ax3.set_title('Confusion Matrix (Top 10 Only)\n(Diagonal = Correct)', fontweight='bold')
    
    # ===== Plot 4: Summary Table =====
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary = "BEST TRIGGER CANDIDATES\n" + "="*50 + "\n\n"
    summary += f"{'Rank':<6}{'Phoneme':<10}{'Accuracy':<12}{'Specificity':<12}{'Score':<10}\n"
    summary += "-"*50 + "\n"
    
    for i, (p, s) in enumerate(top10[:10]):
        summary += f"{i+1:<6}{p:<10}{s['accuracy']:.1f}%{'':<5}{s['specificity']:.2f}%{'':<4}{s['distinctiveness']:.1f}\n"
    
    summary += "\n" + "="*50 + "\n"
    summary += "\nWHY THESE PHONEMES?\n"
    summary += "-"*50 + "\n"
    summary += "- High Accuracy: Model correctly identifies them\n"
    summary += "- High Specificity: Other phonemes NOT confused as these\n"
    summary += "- Low False Positives: Won't trigger accidentally\n"
    
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f8f9fa', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_enhanced.png',
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print("✓ Saved: outputs/confusion_enhanced.png")

if __name__ == '__main__':
    data = load_data()
    scores = calculate_distinctiveness(data)
    plot_enhanced(data, scores)
    
    print("\n🎯 Top 10 Most Distinctive Phonemes (Best Triggers):")
    for i, (p, s) in enumerate(sorted(scores.items(), key=lambda x: -x[1]['distinctiveness'])[:10]):
        print(f"  {i+1}. {p}: {s['distinctiveness']:.2f}% (Acc: {s['accuracy']:.1f}%, Spec: {s['specificity']:.2f}%)")
