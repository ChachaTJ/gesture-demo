"""
CPU Decoder - EXACT 92% CONFIGURATION
Based on your Optuna optimization results
"""
import torch
import numpy as np
import h5py
import scipy.io as sio
import pandas as pd
from pathlib import Path
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(4)

PHONEME_MAP = {
    0: '<blank>', 1: 'AA', 2: 'AE', 3: 'AH', 4: 'AO', 5: 'AW',
    6: 'AY', 7: 'B', 8: 'CH', 9: 'D', 10: 'DH', 11: 'EH', 12: 'ER',
    13: 'EY', 14: 'F', 15: 'G', 16: 'HH', 17: 'IH', 18: 'IY', 19: 'JH',
    20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'NG', 25: 'OW', 26: 'OY',
    27: 'P', 28: 'R', 29: 'S', 30: 'SH', 31: 'T', 32: 'TH', 33: 'UH',
    34: 'UW', 35: 'V', 36: 'W', 37: 'Y', 38: 'Z', 39: 'ZH', 40: 'SIL', 41: 'SP'
}


class CPUPhonemeDecoder:
    """
    Your 92% model - EXACT configuration from Optuna optimization
    
    Settings that achieved 92.14%:
    - LM: outputs/t15_phoneme_lm.pkl (your original LM from training)
    - Beam size: 128
    - LM weight: 1.0  
    - Length penalty: 0.9
    """

    def __init__(self, use_wfst=True):
        print("Loading your 92% model configuration...")

        from model import ConformerXL
        import yaml

        checkpoint = torch.load('outputs/v4_model_1_final.pt', map_location='cpu')
        with open('outputs/v4_model_1_config.yaml') as f:
            config = yaml.safe_load(f)

        self.model = ConformerXL(config)
        
        # Use EMA shadow weights (better accuracy, 0.9997 decay)
        if 'ema_state_dict' in checkpoint and 'shadow' in checkpoint['ema_state_dict']:
            print("   ✓ Loading EMA shadow weights (better accuracy)")
            self.model.load_state_dict(checkpoint['ema_state_dict']['shadow'])
        else:
            print("   ⚠ EMA not found, using regular model weights")
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        print(f"✓ Model loaded: {params / 1e6:.1f}M parameters")
        
        # Initialize ground truth
        self.ground_truth = None
        
        # Load WFST with EXACT 92% settings
        self.use_wfst = use_wfst
        if use_wfst:
            try:
                from wfst_decoder_cpu import WFSTBeamDecoderCPU
                print("Loading WFST decoder with 92% configuration...")
                self.wfst_decoder = WFSTBeamDecoderCPU(
                    lm_path='outputs/t15_phoneme_lm.pkl',  # ← Your original LM!
                    beam_size=128,                          # ← Optuna best
                    lm_weight=1.0,                          # ← Optuna best
                    length_penalty=0.9                      # ← Optuna best
                )
                print("✓ WFST loaded - Expected accuracy: 92.14%")
            except Exception as e:
                print(f"⚠️  Could not load WFST: {e}")
                print("   Falling back to greedy (~89%)")
                self.use_wfst = False
        else:
            print("✓ Greedy mode - Expected accuracy: ~89%")

    def load_file(self, filepath):
        """Load CSV, HDF5, or MAT file"""
        ext = Path(filepath).suffix.lower()
        self.ground_truth = None

        if ext in ['.h5', '.hdf5']:
            with h5py.File(filepath, 'r') as f:
                trial_keys = sorted([k for k in f.keys() if k.startswith('trial_')])
                if trial_keys:
                    first_trial = f[trial_keys[0]]
                    if isinstance(first_trial, h5py.Group):
                        for feat_key in ['input_features', 'neural_features', 'neural_data']:
                            if feat_key in first_trial:
                                data = first_trial[feat_key][:]
                                
                                if 'seq_class_ids' in first_trial:
                                    self.ground_truth = first_trial['seq_class_ids'][:]

                                if len(data.shape) == 2:
                                    if data.shape[0] == 512:
                                        return data.T[np.newaxis, ...]
                                    elif data.shape[1] == 512:
                                        return data[np.newaxis, ...]

        elif ext == '.csv':
            df = pd.read_csv(filepath, header=None)
            data = df.values.astype(np.float32)
            if data.shape[1] == 512:
                return data[np.newaxis, ...]
            elif data.shape[0] == 512:
                return data.T[np.newaxis, ...]

        elif ext == '.mat':
            mat = sio.loadmat(filepath)
            for key in ['neural_data', 'data', 'features']:
                if key in mat:
                    data = mat[key].astype(np.float32)
                    if len(data.shape) == 2 and data.shape[1] == 512:
                        return data[np.newaxis, ...]
                    elif len(data.shape) == 2 and data.shape[0] == 512:
                        return data.T[np.newaxis, ...]

        raise ValueError(f"Could not load neural data from {filepath}")

    def decode(self, neural_data):
        """Decode neural data to phonemes"""
        if len(neural_data.shape) == 2:
            neural_data = neural_data[np.newaxis, ...]

        N, T, C = neural_data.shape
        print(f"Decoding {N} sample(s) with 92% configuration...")

        results = []

        with torch.no_grad():
            for i in range(N):
                neural = torch.tensor(neural_data[i], dtype=torch.float32).unsqueeze(0)
                length = torch.tensor([T])

                logits, _ = self.model(neural, length)

                # Use WFST with optimal settings or greedy fallback
                if self.use_wfst and hasattr(self, 'wfst_decoder'):
                    decoded = self.wfst_decoder.decode(logits)[0]
                else:
                    # Greedy CTC decode
                    predictions = torch.argmax(logits, dim=-1)[0]
                    decoded = []
                    prev = -1
                    for p in predictions:
                        p_val = p.item()
                        if p_val != 0 and p_val != prev:
                            decoded.append(p_val)
                        prev = p_val

                phonemes = [PHONEME_MAP.get(p, f'UNK{p}') for p in decoded]

                result = {
                    'sample_id': i,
                    'phonemes': phonemes,
                    'phoneme_string': ' '.join(phonemes),
                    'num_phonemes': len(phonemes),
                }

                # Calculate accuracy if GT available
                if self.ground_truth is not None:
                    gt_phonemes = [self.ground_truth[j] for j in range(len(self.ground_truth)) 
                                   if self.ground_truth[j] != 0]
                    gt_phoneme_strings = [PHONEME_MAP.get(p, f'UNK{p}') for p in gt_phonemes]

                    # Use edit distance (PER)
                    accuracy = self._calculate_per(decoded, gt_phonemes)
                    correct = int(accuracy * len(gt_phonemes) / 100)

                    result['ground_truth'] = ' '.join(gt_phoneme_strings)
                    result['accuracy'] = accuracy
                    result['correct'] = correct
                    result['total'] = len(gt_phonemes)

                    print(f"  Sample {i + 1}/{N}: {len(phonemes)} phonemes, Accuracy: {accuracy:.1f}%")
                else:
                    print(f"  Sample {i + 1}/{N}: {len(phonemes)} phonemes")

                results.append(result)

        return results

    def _calculate_per(self, pred, target):
        """Calculate PER using edit distance"""
        m, n = len(pred), len(target)
        if n == 0:
            return 0.0
        if m == 0:
            return 0.0
        
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if pred[i-1] == target[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        edit_dist = dp[m][n]
        per = edit_dist / n
        accuracy = max(0, 100 * (1 - per))
        return accuracy


if __name__ == '__main__':
    decoder = CPUPhonemeDecoder()
    print("\n92% decoder ready!")