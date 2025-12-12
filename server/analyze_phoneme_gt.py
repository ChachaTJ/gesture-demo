"""
Phoneme GT Quality Analysis
Compare phoneme GT with expected phonemes from sentence GT
"""
import h5py
import numpy as np
import re

# Correct PHONEME_MAP
PHONEME_MAP = {
    0: '<blank>',
    1: 'AA', 2: 'AE', 3: 'AH', 4: 'AO', 5: 'AW',
    6: 'AY', 7: 'B', 8: 'CH', 9: 'D', 10: 'DH',
    11: 'EH', 12: 'ER', 13: 'EY', 14: 'F', 15: 'G',
    16: 'HH', 17: 'IH', 18: 'IY', 19: 'JH', 20: 'K',
    21: 'L', 22: 'M', 23: 'N', 24: 'NG', 25: 'OW',
    26: 'OY', 27: 'P', 28: 'R', 29: 'S', 30: 'SH',
    31: 'T', 32: 'TH', 33: 'UH', 34: 'UW', 35: 'V',
    36: 'W', 37: 'Y', 38: 'Z', 39: 'ZH', 40: 'SIL', 41: 'SP'
}

# Some expected phoneme patterns for common words
EXPECTED_PHONEMES = {
    "i": ["AY"],
    "will": ["W", "IH", "L"],
    "go": ["G", "OW"],
    "around": ["ER", "AW", "N", "D"],
    "am": ["AE", "M"],
    "talking": ["T", "AO", "K", "IH", "NG"],
    "to": ["T", "UW"],
    "my": ["M", "AY"],
    "family": ["F", "AE", "M", "AH", "L", "IY"],
    "it": ["IH", "T"],
    "is": ["IH", "Z"],
    "the": ["DH", "AH"],
    "why": ["W", "AY"],
    "don't": ["D", "OW", "N", "T"],
    "you": ["Y", "UW"],
    "come": ["K", "AH", "M"],
    "here": ["HH", "IY", "R"],
    "home": ["HH", "OW", "M"],
    "this": ["DH", "IH", "S"],
    "time": ["T", "AY", "M"],
    "what": ["W", "AH", "T"],
    "that": ["DH", "AE", "T"],
    "have": ["HH", "AE", "V"],
    "are": ["AA", "R"],
    "we": ["W", "IY"],
}

def decode_phoneme_gt(seq_ids):
    valid_ids = seq_ids[seq_ids > 0]
    phonemes = [PHONEME_MAP.get(int(p), f'?{p}') for p in valid_ids]
    deduped = []
    prev = None
    for p in phonemes:
        if p != prev:
            deduped.append(p)
        prev = p
    return deduped

def decode_sentence(trans):
    valid = trans[trans != 0]
    return ''.join([chr(int(c)) for c in valid])

def analyze_phoneme_accuracy(sentence, phonemes):
    """Check if phonemes roughly match the sentence"""
    issues = []
    
    # Remove SIL and get phoneme sequence
    phonemes_no_sil = [p for p in phonemes if p != 'SIL']
    
    # Get words from sentence
    words = re.findall(r"[a-zA-Z']+", sentence.lower())
    
    # Check word count vs SIL count (should be roughly words-1 or words SILs)
    sil_count = phonemes.count('SIL')
    word_count = len(words)
    
    if abs(sil_count - word_count) > 2:
        issues.append(f"SIL count ({sil_count}) doesn't match word count ({word_count})")
    
    # Check for expected phonemes in known words
    for word in words:
        if word in EXPECTED_PHONEMES:
            expected = EXPECTED_PHONEMES[word]
            # Check if expected phonemes appear in sequence
            found = all(p in phonemes_no_sil for p in expected)
            if not found:
                issues.append(f"Word '{word}' expected {expected}, some missing in {phonemes_no_sil}")
    
    return issues

# Analyze samples
filepath = '/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final/t15.2023.11.03/data_train.hdf5'

print("=" * 80)
print("PHONEME GROUND TRUTH QUALITY ANALYSIS")
print("=" * 80)

total_issues = 0
total_samples = 0

with h5py.File(filepath, 'r') as f:
    trial_keys = sorted([k for k in f.keys() if k.startswith('trial_')])[:20]  # First 20 trials
    
    for trial_key in trial_keys:
        trial = f[trial_key]
        
        sentence = decode_sentence(trial['transcription'][:])
        phonemes = decode_phoneme_gt(trial['seq_class_ids'][:])
        
        if not sentence:  # Skip empty sentences (like test files)
            continue
            
        total_samples += 1
        issues = analyze_phoneme_accuracy(sentence, phonemes)
        
        print(f"\n{trial_key}:")
        print(f"  Sentence: \"{sentence}\"")
        print(f"  Phonemes: {' '.join(phonemes)}")
        
        if issues:
            total_issues += len(issues)
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("  ✅ Looks OK")

print("\n" + "=" * 80)
print(f"SUMMARY: {total_issues} potential issues found in {total_samples} samples")
print("=" * 80)

# Additional analysis: Check phoneme length consistency
print("\n\nPHONEME LENGTH ANALYSIS:")
with h5py.File(filepath, 'r') as f:
    lengths = []
    for trial_key in sorted([k for k in f.keys() if k.startswith('trial_')])[:50]:
        trial = f[trial_key]
        sentence = decode_sentence(trial['transcription'][:])
        phonemes = decode_phoneme_gt(trial['seq_class_ids'][:])
        
        if sentence:
            char_count = len(sentence.replace(" ", "").replace(".", "").replace(",", ""))
            phoneme_count = len([p for p in phonemes if p != 'SIL'])
            ratio = phoneme_count / char_count if char_count > 0 else 0
            lengths.append((sentence[:30], char_count, phoneme_count, ratio))
    
    print(f"{'Sentence':<32} | Chars | Phon | Ratio")
    print("-" * 60)
    for sent, chars, phons, ratio in lengths[:15]:
        print(f"{sent:<32} | {chars:>5} | {phons:>4} | {ratio:.2f}")
    
    ratios = [x[3] for x in lengths]
    print(f"\nAverage phoneme/char ratio: {np.mean(ratios):.2f} (std: {np.std(ratios):.2f})")
