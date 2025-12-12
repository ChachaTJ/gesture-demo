"""
Deeper Phoneme GT Analysis - Compare with CMU Pronunciation Dictionary
"""
import h5py
import numpy as np
import re

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

# CMU-style expected pronunciations for common words in the dataset
# These are "canonical" pronunciations
CMU_DICT = {
    "i": "AY",
    "will": "W IH L",
    "go": "G OW",
    "around": "ER AW N D",
    "am": "AE M",
    "talking": "T AO K IH NG",
    "to": "T UW",
    "my": "M AY",
    "family": "F AE M AH L IY",
    "it": "IH T",
    "is": "IH Z",
    "looking": "L UH K IH NG",
    "quite": "K W AY T",
    "hard": "HH AA R D",
    "why": "W AY",  # or HW AY
    "don't": "D OW N T",
    "you": "Y UW",
    "come": "K AH M",
    "here": "HH IY R",
    "usually": "Y UW ZH AH W AH L IY",  # complex!
    "home": "HH OW M",
    "by": "B AY",
    "this": "DH IH S",
    "time": "T AY M",
    "please": "P L IY Z",
    "think": "TH IH NG K",
    "through": "TH R UW",
    "came": "K EY M",
    "for": "F AO R",
    "what": "W AH T",  # or HW AH T
    "are": "AA R",
    "doing": "D UW IH NG",
    "right": "R AY T",
    "now": "N AW",
    "say": "S EY",
    "we": "W IY",
    "should": "SH UH D",
    "do": "D UW",
    "the": "DH AH",  # or DH IY, DH AX
    "can't": "K AE N T",
    "see": "S IY",
    "can": "K AE N",
    "both": "B OW TH",
    "of": "AH V",
    "us": "AH S",
    "going": "G OW IH NG",
    "very": "V EH R IY",
    "good": "G UH D",
    "give": "G IH V",
    "a": "AH",
    "call": "K AO L",
    "next": "N EH K S T",
    "week": "W IY K",
    "trying": "T R AY IH NG",
    "same": "S EY M",
    "thing": "TH IH NG",
    "how": "HH AW",
    "about": "AH B AW T",
    "there": "DH EH R",
    "together": "T AH G EH DH ER",
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

def split_by_sil(phonemes):
    """Split phoneme sequence by SIL tokens to get word-level phonemes"""
    words = []
    current = []
    for p in phonemes:
        if p == 'SIL':
            if current:
                words.append(current)
                current = []
        else:
            current.append(p)
    if current:
        words.append(current)
    return words

def compare_word_phonemes(word, actual_phonemes, expected_str):
    """Compare actual phonemes with expected CMU pronunciation"""
    expected = expected_str.split()
    actual_str = ' '.join(actual_phonemes)
    expected_str_clean = ' '.join(expected)
    
    if actual_str == expected_str_clean:
        return "✅ EXACT MATCH"
    
    # Check if it's close (allowing for minor variations)
    actual_set = set(actual_phonemes)
    expected_set = set(expected)
    
    missing = expected_set - actual_set
    extra = actual_set - expected_set
    
    if not missing and not extra:
        return "⚠️ SAME PHONEMES, DIFFERENT ORDER"
    elif len(missing) <= 1 and len(extra) <= 1:
        return f"⚠️ MINOR DIFF: missing={missing}, extra={extra}"
    else:
        return f"❌ MISMATCH: expected={expected_str_clean}, got={actual_str}"

# Analyze
filepath = '/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final/t15.2023.11.03/data_train.hdf5'

print("=" * 100)
print("WORD-BY-WORD PHONEME COMPARISON WITH CMU DICTIONARY")
print("=" * 100)

matches = 0
minor_diffs = 0
mismatches = 0
total_words = 0

with h5py.File(filepath, 'r') as f:
    for i in range(10):  # First 10 trials
        trial = f[f'trial_{i:04d}']
        
        sentence = decode_sentence(trial['transcription'][:])
        phonemes = decode_phoneme_gt(trial['seq_class_ids'][:])
        
        if not sentence:
            continue
        
        words = re.findall(r"[a-zA-Z']+", sentence.lower())
        phoneme_words = split_by_sil(phonemes)
        
        print(f"\n{'='*60}")
        print(f"Sentence: \"{sentence}\"")
        print(f"Words: {words}")
        print(f"Phoneme groups: {len(phoneme_words)}")
        print("-" * 60)
        
        for j, word in enumerate(words):
            if j < len(phoneme_words):
                actual = phoneme_words[j]
                
                if word in CMU_DICT:
                    result = compare_word_phonemes(word, actual, CMU_DICT[word])
                    total_words += 1
                    if "EXACT" in result:
                        matches += 1
                    elif "MINOR" in result or "SAME" in result:
                        minor_diffs += 1
                    else:
                        mismatches += 1
                    print(f"  '{word}': {' '.join(actual)}")
                    print(f"    Expected: {CMU_DICT[word]}")
                    print(f"    {result}")
                else:
                    print(f"  '{word}': {' '.join(actual)} (no CMU reference)")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total words checked: {total_words}")
print(f"  ✅ Exact matches: {matches} ({100*matches/total_words:.1f}%)")
print(f"  ⚠️ Minor differences: {minor_diffs} ({100*minor_diffs/total_words:.1f}%)")
print(f"  ❌ Mismatches: {mismatches} ({100*mismatches/total_words:.1f}%)")
