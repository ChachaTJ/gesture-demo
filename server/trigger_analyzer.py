#!/usr/bin/env python3
"""
Phoneme Trigger Analyzer for BCI/ALS Applications
==================================================
Finds optimal trigger phonemes based on:
  1. High recognition accuracy (from model predictions)
  2. Low daily frequency (from CMU ARPABET corpus)

This combination ensures triggers that are:
  - RELIABLE: Correctly recognized when intended
  - SAFE: Rarely triggered accidentally in normal speech
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from typing import List, Tuple, Dict, Optional

# =============================================================================
# PHONEME FREQUENCY DICTIONARY (LibriSpeech LM Corpus - ACTUAL DATA)
# Source: 5,000,000 lines, 364,713,225 phonemes analyzed via G2P
# Lower values = less common in daily speech = safer as triggers
# =============================================================================
PHONEME_FREQ = {
    # --- [매우 흔함: 빈도 > 3%] ---
    'AH': 0.106221,  # (Schwa) 압도적 1등
    'N':  0.072703,
    'T':  0.070311,
    'IH': 0.058426,
    'D':  0.051900,
    'S':  0.045924,
    'R':  0.043121,
    'L':  0.039745,
    'DH': 0.032499,  # (The, That)
    'IY': 0.031927,

    # --- [보통: 1% ~ 3%] ---
    'AE': 0.029916,
    'M':  0.029070,
    'EH': 0.028032,
    'Z':  0.028014,
    'K':  0.026321,
    'ER': 0.026059,
    'W':  0.022137,
    'HH': 0.021650,
    'V':  0.020049,
    'B':  0.019523,
    'F':  0.019370,
    'UW': 0.018533,
    'P':  0.018192,
    'AA': 0.017892,
    'AY': 0.016896,
    'AO': 0.016641,
    'EY': 0.015277,
    'OW': 0.012610,
    'NG': 0.011137,

    # --- [희귀함: 0.5% ~ 1% (Trigger 후보군)] ---
    'G':  0.008670,
    'SH': 0.007677,  # (She)
    'AW': 0.006492,  # (Out)
    'Y':  0.006224,  # (Yes)
    'CH': 0.005632,  # (Cheese)

    # --- [★ 초-희귀함 (Best Triggers) ★] ---
    'TH': 0.004680,  # (Think)
    'JH': 0.004499,  # (Joy)
    'UH': 0.004488,  # (Book)
    'OY': 0.001099,  # (Boy) - 매우 희귀
    'ZH': 0.000441,  # (Vision) - 가장 희귀! 최고 추천

    # Special tokens (무시)
    'SIL': 0.1, 'SP': 0.1, 'BLANK': 0.1, ' | ': 0.1
}

# Model's phoneme list (matches LOGIT_TO_PHONEME)
MODEL_PHONEMES = [
    'BLANK', 'AA', 'AE', 'AH', 'AO', 'AW',
    'AY', 'B', 'CH', 'D', 'DH',
    'EH', 'ER', 'EY', 'F', 'G',
    'HH', 'IH', 'IY', 'JH', 'K',
    'L', 'M', 'N', 'NG', 'OW',
    'OY', 'P', 'R', 'S', 'SH',
    'T', 'TH', 'UH', 'UW', 'V',
    'W', 'Y', 'Z', 'ZH', ' | '
]


def analyze_and_plot_triggers(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Optional[List[str]] = None,
    top_k: int = 5,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze phoneme predictions and find optimal trigger candidates.
    
    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels (integer indices or strings)
    y_pred : np.ndarray
        Predicted labels (integer indices or strings)
    label_names : List[str], optional
        List of phoneme names. If None, uses MODEL_PHONEMES
    top_k : int
        Number of top trigger candidates to highlight
    save_path : str, optional
        Path to save the visualization
        
    Returns
    -------
    pd.DataFrame
        Analysis results sorted by trigger score (higher = better)
    """
    
    # Use default label names if not provided
    if label_names is None:
        label_names = MODEL_PHONEMES
    
    # Convert to numpy arrays if needed
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    # If labels are integers, map to names
    if np.issubdtype(y_true.dtype, np.integer):
        y_true_names = [label_names[i] for i in y_true]
        y_pred_names = [label_names[i] for i in y_pred]
    else:
        y_true_names = y_true.tolist()
        y_pred_names = y_pred.tolist()
    
    # Get unique phonemes present in data
    unique_phonemes = sorted(set(y_true_names))
    
    # Calculate per-class metrics
    results = []
    for phoneme in unique_phonemes:
        # Skip special tokens
        if phoneme in ['BLANK', 'SIL', 'SP', ' | ']:
            continue
        
        # Binary classification for this phoneme
        y_true_binary = [1 if p == phoneme else 0 for p in y_true_names]
        y_pred_binary = [1 if p == phoneme else 0 for p in y_pred_names]
        
        # Calculate metrics
        tp = sum(t == 1 and p == 1 for t, p in zip(y_true_binary, y_pred_binary))
        fp = sum(t == 0 and p == 1 for t, p in zip(y_true_binary, y_pred_binary))
        fn = sum(t == 1 and p == 0 for t, p in zip(y_true_binary, y_pred_binary))
        tn = sum(t == 0 and p == 0 for t, p in zip(y_true_binary, y_pred_binary))
        
        # Precision, Recall, F1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Specificity (True Negative Rate) - important for avoiding false triggers
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Get frequency (default to 0.01 if not found)
        frequency = PHONEME_FREQ.get(phoneme, 0.01)
        
        # Total samples for this phoneme
        total_samples = sum(y_true_binary)
        
        results.append({
            'phoneme': phoneme,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'frequency': frequency,
            'total_samples': total_samples,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # ==========================================================================
    # TRIGGER SCORE CALCULATION
    # ==========================================================================
    # Formula: Score = Precision * (1 - Frequency) * Recall_weight
    # - High Precision: When model says this phoneme, it's correct
    # - Low Frequency: Rarely occurs in normal speech (safety)
    # - Recall also matters: Should detect when user actually says it
    # ==========================================================================
    
    df['safety_score'] = 1 - df['frequency']  # Lower frequency = higher safety
    df['trigger_score'] = (
        df['precision'] * 0.4 +      # Precision is important
        df['recall'] * 0.3 +         # Should detect when intended
        df['specificity'] * 0.1 +    # Should not trigger when not intended
        df['safety_score'] * 0.2     # Low daily frequency = safe
    )
    
    # Normalize to 0-100
    df['trigger_score'] = df['trigger_score'] * 100
    
    # Sort by trigger score
    df = df.sort_values('trigger_score', ascending=False).reset_index(drop=True)
    
    # ==========================================================================
    # VISUALIZATION
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Phoneme Trigger Analysis for BCI/ALS', fontsize=16, fontweight='bold', y=1.02)
    
    # --- Plot 1: Trigger Score Ranking ---
    ax1 = axes[0, 0]
    colors = ['#7bed9f' if i < top_k else '#a5b1c2' for i in range(len(df))]
    bars = ax1.barh(df['phoneme'][::-1], df['trigger_score'][::-1], color=colors[::-1])
    ax1.set_xlabel('Trigger Score (0-100)', fontweight='bold')
    ax1.set_title(f'🏆 Top {top_k} Trigger Candidates', fontweight='bold')
    ax1.axvline(x=df['trigger_score'].median(), color='red', linestyle='--', label='Median')
    ax1.legend()
    
    # --- Plot 2: Precision vs Frequency (Safety-Accuracy Trade-off) ---
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df['frequency'] * 100,
        df['precision'] * 100,
        s=df['total_samples'] * 2,
        c=df['trigger_score'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    
    # Annotate top phonemes
    for i, row in df.head(top_k).iterrows():
        ax2.annotate(
            row['phoneme'],
            (row['frequency'] * 100, row['precision'] * 100),
            fontsize=10, fontweight='bold',
            xytext=(5, 5), textcoords='offset points'
        )
    
    ax2.set_xlabel('Frequency in Speech (%)', fontweight='bold')
    ax2.set_ylabel('Precision (%)', fontweight='bold')
    ax2.set_title('🎯 Safety vs Accuracy Trade-off\n(Top-Right = Best)', fontweight='bold')
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5, label='90% Precision')
    ax2.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='2% Frequency')
    ax2.legend(loc='lower right')
    plt.colorbar(scatter, ax=ax2, label='Trigger Score')
    
    # Add quadrant labels
    ax2.text(0.5, 95, '✅ IDEAL\n(Safe & Accurate)', fontsize=9, ha='center', color='green', alpha=0.8)
    ax2.text(6, 95, '⚠️ Accurate but\nFrequent', fontsize=9, ha='center', color='orange', alpha=0.8)
    ax2.text(0.5, 75, '⚠️ Safe but\nLess Accurate', fontsize=9, ha='center', color='orange', alpha=0.8)
    
    # --- Plot 3: Precision, Recall, F1 Comparison ---
    ax3 = axes[1, 0]
    top_df = df.head(10)
    x = np.arange(len(top_df))
    width = 0.25
    
    ax3.bar(x - width, top_df['precision'] * 100, width, label='Precision', color='#3498db')
    ax3.bar(x, top_df['recall'] * 100, width, label='Recall', color='#2ecc71')
    ax3.bar(x + width, top_df['f1_score'] * 100, width, label='F1 Score', color='#9b59b6')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_df['phoneme'], rotation=45, ha='right')
    ax3.set_ylabel('Score (%)', fontweight='bold')
    ax3.set_title('📊 Top 10 Phonemes: Precision, Recall, F1', fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0, 110)
    
    # --- Plot 4: Recommended Triggers Summary ---
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create summary text
    top_triggers = df.head(top_k)
    summary_text = "🎯 RECOMMENDED TRIGGER PHONEMES\n" + "="*40 + "\n\n"
    
    for i, row in top_triggers.iterrows():
        emoji = "🥇" if i == 0 else "🥈" if i == 1 else "🥉" if i == 2 else "✅"
        summary_text += (
            f"{emoji} {row['phoneme']:4s}  "
            f"Score: {row['trigger_score']:.1f}  "
            f"Precision: {row['precision']*100:.1f}%  "
            f"Freq: {row['frequency']*100:.2f}%\n"
        )
    
    summary_text += "\n" + "="*40 + "\n"
    summary_text += f"Total Phonemes Analyzed: {len(df)}\n"
    summary_text += f"Average Trigger Score: {df['trigger_score'].mean():.1f}\n"
    summary_text += f"Top {top_k} Average Score: {top_triggers['trigger_score'].mean():.1f}\n"
    
    # Reasoning
    summary_text += "\n📝 WHY THESE PHONEMES?\n"
    summary_text += "─" * 40 + "\n"
    summary_text += "• High Precision → Low false triggers\n"
    summary_text += "• Low Frequency → Rare in normal speech\n"
    summary_text += "• Good Recall → Detects when intended\n"
    
    ax4.text(0.05, 0.95, summary_text, 
             transform=ax4.transAxes, 
             fontsize=11, 
             verticalalignment='top',
             fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.show()
    
    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================
    print("\n" + "="*60)
    print("🎯 OPTIMAL TRIGGER PHONEMES FOR BCI/ALS")
    print("="*60)
    print(f"\n{'Rank':<6}{'Phoneme':<10}{'Score':<10}{'Precision':<12}{'Recall':<10}{'Freq':<10}")
    print("-"*60)
    
    for i, row in top_triggers.iterrows():
        print(f"{i+1:<6}{row['phoneme']:<10}{row['trigger_score']:.1f}{'':>4}"
              f"{row['precision']*100:.1f}%{'':>5}{row['recall']*100:.1f}%{'':>4}"
              f"{row['frequency']*100:.2f}%")
    
    print("\n" + "="*60)
    
    return df


# =============================================================================
# LOAD FROM REAL MODEL DATA (confusion_matrix.json)
# =============================================================================
def analyze_from_confusion_json(
    json_path: str = '/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_matrix.json',
    top_k: int = 5,
    save_path: Optional[str] = None
) -> pd.DataFrame:
    """
    Analyze trigger candidates from existing confusion matrix JSON.
    
    This uses the REAL model performance data from confusion_matrix.json
    instead of synthetic demo data.
    """
    import json
    
    print(f"Loading real model data from: {json_path}")
    
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cm = data.get('confusion_matrix', {})
    totals = data.get('total_samples_per_phoneme', {})
    
    results = []
    
    for phoneme in cm.keys():
        # Skip special tokens
        if phoneme in ['BLANK', 'SIL', 'SP', ' | ']:
            continue
        
        # Get confusion data for this phoneme
        predictions = cm[phoneme]
        total = totals.get(phoneme, 0)
        
        if total == 0:
            continue
        
        # True Positives: correct predictions
        tp = predictions.get(phoneme, 0)
        
        # False Negatives: should be this phoneme but predicted as something else
        fn = total - tp
        
        # False Positives: other phonemes incorrectly predicted as this phoneme
        fp = 0
        for other_phoneme, other_preds in cm.items():
            if other_phoneme != phoneme:
                fp += other_preds.get(phoneme, 0)
        
        # Calculate total samples for TN calculation
        total_all = sum(totals.values())
        tn = total_all - tp - fp - fn
        
        # Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # Frequency from dictionary
        frequency = PHONEME_FREQ.get(phoneme, 0.01)
        
        results.append({
            'phoneme': phoneme,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'frequency': frequency,
            'total_samples': total,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn
        })
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Calculate trigger score
    df['safety_score'] = 1 - df['frequency']
    df['trigger_score'] = (
        df['precision'] * 0.4 +
        df['recall'] * 0.3 +
        df['specificity'] * 0.1 +
        df['safety_score'] * 0.2
    ) * 100
    
    # Sort by trigger score
    df = df.sort_values('trigger_score', ascending=False).reset_index(drop=True)
    
    # ==========================================================================
    # VISUALIZATION (same as before)
    # ==========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Phoneme Trigger Analysis (REAL MODEL DATA)', fontsize=16, fontweight='bold', y=1.02)
    
    # Plot 1: Trigger Score Ranking
    ax1 = axes[0, 0]
    colors = ['#7bed9f' if i < top_k else '#a5b1c2' for i in range(len(df))]
    ax1.barh(df['phoneme'][::-1], df['trigger_score'][::-1], color=colors[::-1])
    ax1.set_xlabel('Trigger Score (0-100)', fontweight='bold')
    ax1.set_title(f'Top {top_k} Trigger Candidates', fontweight='bold')
    ax1.axvline(x=df['trigger_score'].median(), color='red', linestyle='--', label='Median')
    ax1.legend()
    
    # Plot 2: Precision vs Frequency
    ax2 = axes[0, 1]
    scatter = ax2.scatter(
        df['frequency'] * 100,
        df['precision'] * 100,
        s=df['total_samples'] * 0.5,
        c=df['trigger_score'],
        cmap='RdYlGn',
        alpha=0.7,
        edgecolors='black'
    )
    
    for i, row in df.head(top_k).iterrows():
        ax2.annotate(
            row['phoneme'],
            (row['frequency'] * 100, row['precision'] * 100),
            fontsize=10, fontweight='bold',
            xytext=(5, 5), textcoords='offset points'
        )
    
    ax2.set_xlabel('Frequency in Speech (%)', fontweight='bold')
    ax2.set_ylabel('Precision (%)', fontweight='bold')
    ax2.set_title('Safety vs Accuracy Trade-off', fontweight='bold')
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.5)
    ax2.axvline(x=1, color='red', linestyle='--', alpha=0.5)
    plt.colorbar(scatter, ax=ax2, label='Trigger Score')
    
    # Plot 3: Precision, Recall, F1
    ax3 = axes[1, 0]
    top_df = df.head(10)
    x = np.arange(len(top_df))
    width = 0.25
    
    ax3.bar(x - width, top_df['precision'] * 100, width, label='Precision', color='#3498db')
    ax3.bar(x, top_df['recall'] * 100, width, label='Recall', color='#2ecc71')
    ax3.bar(x + width, top_df['f1_score'] * 100, width, label='F1 Score', color='#9b59b6')
    
    ax3.set_xticks(x)
    ax3.set_xticklabels(top_df['phoneme'], rotation=45, ha='right')
    ax3.set_ylabel('Score (%)', fontweight='bold')
    ax3.set_title('Top 10: Precision, Recall, F1', fontweight='bold')
    ax3.legend()
    ax3.set_ylim(0, 110)
    
    # Plot 4: Summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    top_triggers = df.head(top_k)
    summary = "RECOMMENDED TRIGGER PHONEMES (REAL DATA)\n" + "="*45 + "\n\n"
    
    for idx, row in enumerate(top_triggers.itertuples()):
        rank = ["1st", "2nd", "3rd", "4th", "5th"][idx]
        summary += (
            f"{rank}: {row.phoneme:4s}  "
            f"Score: {row.trigger_score:.1f}  "
            f"Prec: {row.precision*100:.1f}%  "
            f"Freq: {row.frequency*100:.2f}%\n"
        )
    
    summary += "\n" + "="*45 + "\n"
    summary += f"Total Phonemes: {len(df)}\n"
    summary += f"Avg Score: {df['trigger_score'].mean():.1f}\n"
    summary += f"Best Score: {df['trigger_score'].max():.1f}\n"
    
    ax4.text(0.05, 0.95, summary, 
             transform=ax4.transAxes, fontsize=11, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='#f0f0f0', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"✓ Saved visualization to: {save_path}")
    
    plt.show()
    
    # Print results
    print("\n" + "="*65)
    print("🎯 OPTIMAL TRIGGER PHONEMES (REAL MODEL DATA)")
    print("="*65)
    print(f"\n{'Rank':<6}{'Phoneme':<10}{'Score':<10}{'Precision':<12}{'Recall':<10}{'Freq':<10}")
    print("-"*65)
    
    for idx, row in enumerate(top_triggers.itertuples()):
        print(f"{idx+1:<6}{row.phoneme:<10}{row.trigger_score:.1f}{'':>4}"
              f"{row.precision*100:.1f}%{'':>5}{row.recall*100:.1f}%{'':>4}"
              f"{row.frequency*100:.2f}%")
    
    print("\n" + "="*65)
    
    return df


if __name__ == '__main__':
    # Run with REAL model data!
    results = analyze_from_confusion_json(
        json_path='/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_matrix.json',
        top_k=5,
        save_path='/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/trigger_analysis_real.png'
    )
    
    # Save results to CSV
    results.to_csv('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/trigger_analysis_real.csv', index=False)
    print("\n✓ Saved results to outputs/trigger_analysis_real.csv")

