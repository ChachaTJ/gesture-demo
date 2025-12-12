"""
PER→WER Evaluation Script
=========================
1. Load validation HDF5 files (_val suffix) from data directory
2. Run CPUPhonemeDecoder inference
3. Calculate PER (Phoneme Error Rate)
4. Use LLM (Anthropic Claude) to convert phonemes to sentences
5. Calculate WER (Word Error Rate)

Usage:
    python evaluate_per_wer.py --data_dir /path/to/hdf5_data_final
"""

import os
import sys
import argparse
import h5py
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import time

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cpu_decoder import CPUPhonemeDecoder, PHONEME_MAP

# Anthropic API key from api_server.py
ANTHROPIC_API_KEY = 'INSERT_API_KEY'


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


def calculate_per(predicted_phonemes, gt_phonemes):
    """Calculate Phoneme Error Rate."""
    pred_list = predicted_phonemes.split()
    gt_list = gt_phonemes.split()
    
    # Filter out silence tokens for fair comparison
    pred_list = [p for p in pred_list if p not in ['SIL', 'SP', '<blank>']]
    gt_list = [p for p in gt_list if p not in ['SIL', 'SP', '<blank>']]
    
    if len(gt_list) == 0:
        return 0.0 if len(pred_list) == 0 else 1.0
    
    distance = levenshtein_distance(pred_list, gt_list)
    return distance / len(gt_list)


def calculate_wer(predicted_sentence, gt_sentence):
    """Calculate Word Error Rate."""
    pred_words = predicted_sentence.lower().strip().split()
    gt_words = gt_sentence.lower().strip().split()
    
    if len(gt_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    
    distance = levenshtein_distance(pred_words, gt_words)
    return distance / len(gt_words)


def decode_ascii_transcription(ascii_codes):
    """Convert numpy array of ASCII codes to string."""
    try:
        valid_codes = ascii_codes[ascii_codes != 0]
        chars = [chr(int(c)) for c in valid_codes]
        return "".join(chars).strip()
    except:
        return ""


def decode_phoneme_gt(seq_class_ids):
    """Convert seq_class_ids to phoneme string."""
    try:
        valid_ids = seq_class_ids[seq_class_ids > 0]
        phonemes = [PHONEME_MAP.get(int(p), f'?{p}') for p in valid_ids]
        # Remove consecutive duplicates
        deduped = []
        prev = None
        for p in phonemes:
            if p != prev:
                deduped.append(p)
            prev = p
        return ' '.join(deduped)
    except:
        return ""


def phonemes_to_sentence_llm(phonemes, api_key):
    """Use Claude to convert phonemes to sentence."""
    import anthropic
    
    client = anthropic.Anthropic(api_key=api_key)
    
    prompt = f'''You are a phoneme-to-text decoder for a brain-computer interface.

PHONEME SEQUENCE (ARPAbet):
{phonemes}

Convert this phoneme sequence to the most likely English sentence.
Return ONLY the sentence, nothing else. No quotes, no explanation.'''
    
    try:
        message = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text.strip()
    except Exception as e:
        print(f"  LLM Error: {e}")
        return ""


def find_validation_files(data_dir):
    """Find all _val.h5 files in data directory."""
    data_path = Path(data_dir)
    val_files = []
    
    # Search in date-based subdirectories
    for subdir in sorted(data_path.iterdir()):
        if subdir.is_dir():
            for f in subdir.glob("*_val.h5"):
                val_files.append(f)
            for f in subdir.glob("*_val.hdf5"):
                val_files.append(f)
    
    # Also check root directory
    for f in data_path.glob("*_val.h5"):
        val_files.append(f)
    for f in data_path.glob("*_val.hdf5"):
        val_files.append(f)
    
    return sorted(set(val_files))


def main():
    parser = argparse.ArgumentParser(description='Evaluate PER and WER on validation data')
    parser.add_argument('--data_dir', type=str, 
                        default='/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final',
                        help='Path to hdf5_data_final directory')
    parser.add_argument('--output', type=str, default='outputs/per_wer_results.json',
                        help='Output file for results')
    parser.add_argument('--max_trials', type=int, default=None,
                        help='Max trials to process (for testing)')
    parser.add_argument('--skip_llm', action='store_true',
                        help='Skip LLM sentence generation (only calculate PER)')
    args = parser.parse_args()
    
    print("=" * 60)
    print("PER→WER Evaluation Pipeline")
    print("=" * 60)
    
    # Find validation files
    print(f"\n📂 Searching for validation files in: {args.data_dir}")
    val_files = find_validation_files(args.data_dir)
    print(f"   Found {len(val_files)} validation files")
    
    if not val_files:
        print("❌ No validation files found!")
        return
    
    for f in val_files:
        print(f"   - {f.name}")
    
    # Load decoder
    print("\n🔧 Loading CPUPhonemeDecoder...")
    decoder = CPUPhonemeDecoder()
    
    # Process all files
    all_results = []
    total_per = []
    total_wer = []
    
    trial_count = 0
    
    for val_file in val_files:
        print(f"\n📄 Processing: {val_file.name}")
        
        with h5py.File(val_file, 'r') as f:
            trial_keys = sorted([k for k in f.keys() if k.startswith('trial_')])
            print(f"   {len(trial_keys)} trials found")
            
            for trial_key in trial_keys:
                if args.max_trials and trial_count >= args.max_trials:
                    break
                
                trial_grp = f[trial_key]
                
                # Get neural data
                if 'input_features' not in trial_grp:
                    continue
                neural_data = trial_grp['input_features'][:]
                
                # Handle shape
                if len(neural_data.shape) == 2:
                    if neural_data.shape[1] == 512:
                        neural_data = neural_data[np.newaxis, ...]
                    elif neural_data.shape[0] == 512:
                        neural_data = neural_data.T[np.newaxis, ...]
                
                # Get ground truths
                gt_sentence = ""
                if 'transcription' in trial_grp:
                    gt_sentence = decode_ascii_transcription(trial_grp['transcription'][()])
                
                gt_phonemes = ""
                if 'seq_class_ids' in trial_grp:
                    gt_phonemes = decode_phoneme_gt(trial_grp['seq_class_ids'][()])
                
                # Skip if no ground truth
                if not gt_sentence and not gt_phonemes:
                    continue
                
                # Decode phonemes
                decoded = decoder.decode(neural_data)
                pred_phonemes = decoded[0]['phoneme_string']
                
                # Calculate PER
                per = calculate_per(pred_phonemes, gt_phonemes) if gt_phonemes else None
                if per is not None:
                    total_per.append(per)
                
                # LLM sentence generation
                pred_sentence = ""
                wer = None
                if not args.skip_llm and gt_sentence:
                    print(f"   [{trial_key}] Calling LLM...")
                    pred_sentence = phonemes_to_sentence_llm(pred_phonemes, ANTHROPIC_API_KEY)
                    wer = calculate_wer(pred_sentence, gt_sentence)
                    total_wer.append(wer)
                    time.sleep(0.5)  # Rate limiting
                
                result = {
                    'file': val_file.name,
                    'trial': trial_key,
                    'gt_sentence': gt_sentence,
                    'gt_phonemes': gt_phonemes,
                    'pred_phonemes': pred_phonemes,
                    'pred_sentence': pred_sentence,
                    'per': per,
                    'wer': wer
                }
                all_results.append(result)
                
                # Print progress
                per_str = f"PER={per:.2%}" if per is not None else "PER=N/A"
                wer_str = f"WER={wer:.2%}" if wer is not None else ""
                print(f"   [{trial_key}] {per_str} {wer_str}")
                if gt_sentence:
                    print(f"      GT: {gt_sentence[:60]}...")
                if pred_sentence:
                    print(f"      Pred: {pred_sentence[:60]}...")
                
                trial_count += 1
        
        if args.max_trials and trial_count >= args.max_trials:
            print(f"\n⚠️ Stopped at max_trials={args.max_trials}")
            break
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION RESULTS")
    print("=" * 60)
    
    avg_per = np.mean(total_per) if total_per else None
    avg_wer = np.mean(total_wer) if total_wer else None
    
    print(f"\nTotal trials evaluated: {len(all_results)}")
    print(f"Average PER: {avg_per:.2%}" if avg_per is not None else "Average PER: N/A")
    print(f"Average WER: {avg_wer:.2%}" if avg_wer is not None else "Average WER: N/A (LLM skipped)")
    
    # Save results
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    summary = {
        'timestamp': datetime.now().isoformat(),
        'data_dir': args.data_dir,
        'total_trials': len(all_results),
        'avg_per': float(avg_per) if avg_per is not None else None,
        'avg_wer': float(avg_wer) if avg_wer is not None else None,
        'results': all_results
    }
    
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n💾 Results saved to: {args.output}")


if __name__ == '__main__':
    main()
