"""
Validation PER Evaluation Script (No LLM)
==========================================
Uses train_val_trials.json to load correct validation trials.
Calculates PER against phoneme ground truth (seq_class_ids).

Usage:
    python evaluate_val_per.py
"""

import os
import sys
import json
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpu_decoder import CPUPhonemeDecoder, PHONEME_MAP


def levenshtein_distance(s1, s2):
    """Calculate Levenshtein (edit) distance between two sequences."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def decode_phoneme_gt(seq_class_ids):
    """Convert seq_class_ids (numpy array) to list of phoneme IDs.
    Removes blanks (0), SIL (40), SP (41) for fair comparison.
    Also removes consecutive duplicates.
    """
    # Filter out padding (0 or negative)
    valid_ids = seq_class_ids[seq_class_ids > 0]
    
    # Remove consecutive duplicates (CTC-style)
    deduped = []
    prev = None
    for p in valid_ids:
        p_int = int(p)
        if p_int != prev:
            deduped.append(p_int)
        prev = p_int
    
    return deduped


def phoneme_ids_to_string(phoneme_ids):
    """Convert list of phoneme IDs to string for display."""
    return ' '.join([PHONEME_MAP.get(p, f'?{p}') for p in phoneme_ids])


def calculate_per(pred_ids, gt_ids, exclude_silence=True):
    """Calculate Phoneme Error Rate.
    
    Args:
        pred_ids: List of predicted phoneme IDs
        gt_ids: List of ground truth phoneme IDs
        exclude_silence: If True, remove SIL (40) and SP (41) from both
    """
    if exclude_silence:
        pred_ids = [p for p in pred_ids if p not in [40, 41]]
        gt_ids = [p for p in gt_ids if p not in [40, 41]]
    
    if len(gt_ids) == 0:
        return 0.0 if len(pred_ids) == 0 else 1.0
    
    distance = levenshtein_distance(pred_ids, gt_ids)
    return distance / len(gt_ids)


def main():
    print("=" * 70)
    print("VALIDATION PER EVALUATION (No LLM)")
    print("=" * 70)
    
    # Paths
    trials_json_path = "/Users/chayoonmin/Downloads/nejm-brain-to-text-main/model_training/trained_models/baseline_rnn/train_val_trials.json"
    data_base_path = "/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final"
    output_path = "outputs/val_per_results.json"
    
    # Load trial definitions
    print(f"\n📂 Loading trial definitions from: {trials_json_path}")
    with open(trials_json_path, 'r') as f:
        trials_data = json.load(f)
    
    val_sessions = trials_data['val']
    print(f"   Found {len(val_sessions)} validation sessions")
    
    # Count total trials
    total_trials = sum(len(s['trials']) for s in val_sessions.values())
    print(f"   Total validation trials: {total_trials}")
    
    # Load decoder
    print("\n🔧 Loading CPUPhonemeDecoder...")
    decoder = CPUPhonemeDecoder()
    
    # Process all validation sessions
    all_results = []
    total_per_values = []
    session_per_values = {}
    
    print("\n" + "=" * 70)
    print("PROCESSING VALIDATION SESSIONS")
    print("=" * 70)
    
    for session_idx, session_info in val_sessions.items():
        trial_indices = session_info['trials']
        session_path_relative = session_info['session_path']
        
        # Skip empty sessions
        if len(trial_indices) == 0:
            continue
        
        # Construct absolute path
        # session_path is relative like "../data/hdf5_data_final/t15.2023.08.13/data_val.hdf5"
        # Extract the date folder
        path_parts = session_path_relative.split('/')
        date_folder = path_parts[-2]  # e.g., "t15.2023.08.13"
        filename = path_parts[-1]  # e.g., "data_val.hdf5"
        
        hdf5_path = os.path.join(data_base_path, date_folder, filename)
        
        if not os.path.exists(hdf5_path):
            print(f"⚠️ File not found: {hdf5_path}")
            continue
        
        print(f"\n📄 Session {session_idx}: {date_folder} ({len(trial_indices)} trials)")
        
        session_pers = []
        
        with h5py.File(hdf5_path, 'r') as f:
            for trial_idx in trial_indices:
                trial_key = f'trial_{trial_idx:04d}'
                
                if trial_key not in f:
                    continue
                
                trial_grp = f[trial_key]
                
                # Get neural data
                if 'input_features' not in trial_grp:
                    continue
                neural_data = trial_grp['input_features'][:]
                
                # Handle shape: [T, 512] or [512, T]
                if len(neural_data.shape) == 2:
                    if neural_data.shape[1] == 512:
                        neural_data = neural_data[np.newaxis, ...]  # [1, T, 512]
                    elif neural_data.shape[0] == 512:
                        neural_data = neural_data.T[np.newaxis, ...]  # [1, T, 512]
                
                # Get ground truth phonemes
                if 'seq_class_ids' not in trial_grp:
                    continue
                
                gt_raw = trial_grp['seq_class_ids'][()]
                gt_ids = decode_phoneme_gt(gt_raw)
                
                if len(gt_ids) == 0:
                    continue
                
                # Decode with model
                decoded = decoder.decode(neural_data)
                
                # Get predicted phoneme IDs (not strings)
                # decoded[0]['phonemes'] is list of strings, convert back to IDs
                pred_phoneme_strings = decoded[0]['phonemes']
                
                # Convert string phonemes to IDs
                str_to_id = {v: k for k, v in PHONEME_MAP.items()}
                pred_ids = []
                for p_str in pred_phoneme_strings:
                    if p_str in str_to_id:
                        pred_ids.append(str_to_id[p_str])
                
                # Calculate PER
                per = calculate_per(pred_ids, gt_ids, exclude_silence=True)
                total_per_values.append(per)
                session_pers.append(per)
                
                result = {
                    'session': session_idx,
                    'date': date_folder,
                    'trial': trial_key,
                    'gt_phonemes': phoneme_ids_to_string(gt_ids),
                    'pred_phonemes': decoded[0]['phoneme_string'],
                    'gt_len': len([p for p in gt_ids if p not in [40, 41]]),
                    'pred_len': len([p for p in pred_ids if p not in [40, 41]]),
                    'per': per
                }
                all_results.append(result)
        
        if session_pers:
            session_avg = np.mean(session_pers)
            session_per_values[session_idx] = session_avg
            print(f"   → Avg PER: {session_avg:.2%} ({len(session_pers)} trials)")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 EVALUATION RESULTS")
    print("=" * 70)
    
    avg_per = np.mean(total_per_values) if total_per_values else None
    
    print(f"\nTotal trials evaluated: {len(all_results)}")
    print(f"Average PER: {avg_per:.2%}" if avg_per is not None else "Average PER: N/A")
    print(f"Accuracy (100-PER): {(1-avg_per)*100:.2f}%" if avg_per is not None else "")
    
    # Per-session breakdown
    print("\n📈 Per-Session PER:")
    sorted_sessions = sorted(session_per_values.items(), key=lambda x: x[1])
    for sess_id, per in sorted_sessions[:10]:
        date = val_sessions[sess_id]['session_path'].split('/')[-2]
        print(f"   Session {sess_id} ({date}): {per:.2%}")
    if len(sorted_sessions) > 10:
        print(f"   ... and {len(sorted_sessions) - 10} more sessions")
    
    # Save results
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_trials': len(all_results),
        'avg_per': float(avg_per) if avg_per is not None else None,
        'accuracy': float((1-avg_per)*100) if avg_per is not None else None,
        'per_session': {k: float(v) for k, v in session_per_values.items()},
        'results': all_results
    }
    
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
