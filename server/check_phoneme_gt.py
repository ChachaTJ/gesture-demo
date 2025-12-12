import h5py
import numpy as np

# Check seq_class_ids - this should be the phoneme ground truth
filepath = "/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final/t15.2023.08.11/data_train.hdf5"

# Phoneme map from cpu_decoder.py
PHONEME_MAP = {
    1: 'AA', 2: 'AE', 3: 'AH', 4: 'AO', 5: 'AW',
    6: 'AY', 7: 'B', 8: 'CH', 9: 'D', 10: 'DH',
    11: 'EH', 12: 'ER', 13: 'EY', 14: 'F', 15: 'G',
    16: 'HH', 17: 'IH', 18: 'IY', 19: 'JH', 20: 'K',
    21: 'L', 22: 'M', 23: 'N', 24: 'NG', 25: 'OW',
    26: 'OY', 27: 'P', 28: 'R', 29: 'S', 30: 'SH',
    31: 'SIL', 32: 'T', 33: 'TH', 34: 'UH', 35: 'UW',
    36: 'V', 37: 'W', 38: 'Y', 39: 'Z', 40: 'ZH'
}

with h5py.File(filepath, 'r') as f:
    trial = f['trial_0000']
    
    print("=== seq_class_ids ===")
    seq_ids = trial['seq_class_ids'][:]
    print(f"Shape: {seq_ids.shape}")
    print(f"Unique values: {np.unique(seq_ids)}")
    print(f"First 50: {seq_ids[:50]}")
    
    # Convert to phoneme string
    valid_ids = seq_ids[seq_ids > 0]  # Remove padding/blank
    phonemes = [PHONEME_MAP.get(int(p), f'?{p}') for p in valid_ids if p > 0]
    # Remove consecutive duplicates (CTC-style)
    deduped = []
    prev = None
    for p in phonemes:
        if p != prev:
            deduped.append(p)
        prev = p
    print(f"\nPhoneme GT: {' '.join(deduped)}")
    
    print(f"\n=== transcription (sentence GT) ===")
    trans = trial['transcription'][:]
    valid = trans[trans != 0]
    sentence = ''.join([chr(int(c)) for c in valid])
    print(f"Sentence GT: {sentence}")
