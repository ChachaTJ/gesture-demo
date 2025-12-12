"""
Phoneme Confusion Matrix Generator
Compares model predictions with ground truth phonemes to identify confusion patterns.
"""
import h5py
import numpy as np
import sys
import os
from collections import defaultdict
import json

sys.path.insert(0, '/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder')
from cpu_decoder import CPUPhonemeDecoder, PHONEME_MAP

# Reverse map: phoneme string -> id
PHONEME_TO_ID = {v: k for k, v in PHONEME_MAP.items()}

# All phonemes (excluding blank)
ALL_PHONEMES = [PHONEME_MAP[i] for i in range(1, 42) if i in PHONEME_MAP]

def decode_phoneme_gt(seq_ids):
    """Convert seq_class_ids to phoneme list (no deduplication for alignment)."""
    valid_ids = seq_ids[seq_ids > 0]
    phonemes = [PHONEME_MAP.get(int(p), f'?{p}') for p in valid_ids]
    # Remove consecutive duplicates
    deduped = []
    prev = None
    for p in phonemes:
        if p != prev:
            deduped.append(p)
        prev = p
    return deduped

def align_sequences(gt_phonemes, pred_phonemes):
    """
    Simple alignment using edit distance with substitution tracking.
    Returns list of (gt_phoneme, pred_phoneme) pairs.
    """
    # Use dynamic programming for alignment
    m, n = len(gt_phonemes), len(pred_phonemes)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    # Initialize
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if gt_phonemes[i-1] == pred_phonemes[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],      # deletion
                    dp[i][j-1],      # insertion
                    dp[i-1][j-1]     # substitution
                )
    
    # Backtrack to get alignment
    alignments = []
    i, j = m, n
    
    while i > 0 or j > 0:
        if i > 0 and j > 0 and gt_phonemes[i-1] == pred_phonemes[j-1]:
            alignments.append((gt_phonemes[i-1], pred_phonemes[j-1]))
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            # Substitution
            alignments.append((gt_phonemes[i-1], pred_phonemes[j-1]))
            i -= 1
            j -= 1
        elif j > 0 and (i == 0 or dp[i][j] == dp[i][j-1] + 1):
            # Insertion (pred has extra)
            alignments.append(('<INS>', pred_phonemes[j-1]))
            j -= 1
        else:
            # Deletion (gt has extra)
            alignments.append((gt_phonemes[i-1], '<DEL>'))
            i -= 1
    
    alignments.reverse()
    return alignments

def build_confusion_matrix(data_path, max_trials=None):
    """Build confusion matrix from predictions."""
    
    # Initialize confusion counts
    confusion = defaultdict(lambda: defaultdict(int))
    total_gt = defaultdict(int)
    
    decoder = CPUPhonemeDecoder()
    
    print(f"Processing {data_path}...")
    
    with h5py.File(data_path, 'r') as f:
        trial_keys = sorted([k for k in f.keys() if k.startswith('trial_')])
        
        if max_trials:
            trial_keys = trial_keys[:max_trials]
        
        for trial_key in trial_keys:
            trial = f[trial_key]
            
            # Skip if no GT
            if 'seq_class_ids' not in trial:
                continue
            
            # Get GT phonemes
            gt_phonemes = decode_phoneme_gt(trial['seq_class_ids'][:])
            
            # Get predictions
            neural_data = trial['input_features'][:]
            if len(neural_data.shape) == 2:
                neural_data = neural_data[np.newaxis, ...]
            
            results = decoder.decode(neural_data)
            pred_phonemes = results[0]['phoneme_string'].split()
            
            # Align and count
            alignments = align_sequences(gt_phonemes, pred_phonemes)
            
            for gt, pred in alignments:
                if gt != '<INS>' and pred != '<DEL>':
                    confusion[gt][pred] += 1
                    total_gt[gt] += 1
    
    return confusion, total_gt

def analyze_confusion(confusion, total_gt, top_k=20):
    """Analyze confusion matrix and find top confused pairs."""
    
    # Calculate confusion percentages
    confused_pairs = []
    
    for gt_phoneme, predictions in confusion.items():
        total = total_gt[gt_phoneme]
        if total == 0:
            continue
        
        for pred_phoneme, count in predictions.items():
            if gt_phoneme != pred_phoneme:  # Only errors
                percentage = (count / total) * 100
                confused_pairs.append({
                    'gt': gt_phoneme,
                    'pred': pred_phoneme,
                    'count': count,
                    'total': total,
                    'percentage': percentage
                })
    
    # Sort by percentage
    confused_pairs.sort(key=lambda x: x['percentage'], reverse=True)
    
    return confused_pairs[:top_k]

def print_confusion_matrix(confusion, total_gt, phonemes_to_show=None):
    """Print confusion matrix as a table."""
    
    if phonemes_to_show is None:
        # Get phonemes with most data
        phonemes_to_show = sorted(total_gt.keys(), key=lambda x: total_gt[x], reverse=True)[:15]
    
    # Header
    print("\nConfusion Matrix (rows=GT, cols=Pred, values=%):")
    print("     ", end="")
    for p in phonemes_to_show:
        print(f"{p:>5}", end="")
    print()
    
    # Rows
    for gt in phonemes_to_show:
        print(f"{gt:>4} ", end="")
        total = total_gt[gt]
        for pred in phonemes_to_show:
            count = confusion[gt][pred]
            if total > 0:
                pct = (count / total) * 100
                if gt == pred:
                    print(f"{pct:>4.0f}%", end="")
                elif pct > 5:
                    print(f"{pct:>4.1f}%", end="")
                else:
                    print(f"    -", end="")
            else:
                print(f"    -", end="")
        print()

if __name__ == '__main__':
    import glob
    
    # Find training files
    train_files = glob.glob('/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final/**/data_train.hdf5', recursive=True)
    
    print("=" * 80)
    print("PHONEME CONFUSION MATRIX ANALYSIS")
    print("=" * 80)
    
    # Aggregate confusion from multiple files
    total_confusion = defaultdict(lambda: defaultdict(int))
    total_gt_counts = defaultdict(int)
    
    for filepath in train_files:  # ALL files
        print(f"\nProcessing: {filepath}")
        confusion, gt_counts = build_confusion_matrix(filepath, max_trials=None)  # ALL trials
        
        # Merge
        for gt, preds in confusion.items():
            for pred, count in preds.items():
                total_confusion[gt][pred] += count
        for gt, count in gt_counts.items():
            total_gt_counts[gt] += count
    
    # Analyze
    print("\n" + "=" * 80)
    print("TOP CONFUSED PHONEME PAIRS")
    print("=" * 80)
    
    top_pairs = analyze_confusion(total_confusion, total_gt_counts, top_k=30)
    
    print(f"\n{'GT':<6} → {'Pred':<6} | {'Count':>6} / {'Total':>6} | {'Error %':>7}")
    print("-" * 50)
    
    for pair in top_pairs:
        print(f"{pair['gt']:<6} → {pair['pred']:<6} | {pair['count']:>6} / {pair['total']:>6} | {pair['percentage']:>6.1f}%")
    
    # Print matrix for common phonemes
    print_confusion_matrix(total_confusion, total_gt_counts)
    
    # Save results
    output = {
        'top_confused_pairs': top_pairs,
        'total_samples_per_phoneme': dict(total_gt_counts),
        'confusion_matrix': {gt: dict(preds) for gt, preds in total_confusion.items()}
    }
    
    with open('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_matrix.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n✓ Saved to outputs/confusion_matrix.json")
