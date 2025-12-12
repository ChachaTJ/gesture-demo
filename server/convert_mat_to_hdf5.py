"""
Convert brain-to-voice MAT files to HDF5 format compatible with phoneme decoder.
"""
import scipy.io as sio
import h5py
import numpy as np
import os
import glob
from pathlib import Path

def preprocess_mat_to_hdf5(mat_path, output_path=None):
    """
    Convert brain-to-voice MAT file to phoneme decoder HDF5 format.
    
    Args:
        mat_path: Path to input .mat file
        output_path: Path for output .hdf5 file (optional, auto-generated if not provided)
    
    Returns:
        Path to created HDF5 file
    """
    # Load MAT file
    print(f"Loading {mat_path}...")
    mat_data = sio.loadmat(mat_path)
    
    # Validate format
    if 'spikepow_trials' not in mat_data or 'threshcross_trials' not in mat_data:
        raise ValueError("Not a valid brain-to-voice MAT file (missing spikepow_trials/threshcross_trials)")
    
    # Generate output path
    if output_path is None:
        output_path = str(Path(mat_path).with_suffix('.hdf5'))
    
    num_trials = mat_data['spikepow_trials'].shape[1]
    sentences = mat_data.get('sentences', ['' for _ in range(num_trials)])
    
    print(f"Converting {num_trials} trials...")
    
    with h5py.File(output_path, 'w') as f:
        for trial_idx in range(num_trials):
            spikepow = mat_data['spikepow_trials'][0, trial_idx]      # (T, 256)
            threshcross = mat_data['threshcross_trials'][0, trial_idx]  # (T, 256)
            
            # Combine: threshcross first (0-255), spikepow second (256-511)
            combined = np.concatenate([threshcross, spikepow], axis=1)  # (T, 512)
            
            # Downsample: 10ms -> 20ms (average every 2 frames)
            T = combined.shape[0] // 2
            downsampled = np.zeros((T, 512), dtype=np.float32)
            for i in range(T):
                downsampled[i] = (combined[2*i] + combined[2*i + 1]) / 2
            
            # Z-score normalize per channel
            input_features = (downsampled - downsampled.mean(axis=0)) / (downsampled.std(axis=0) + 1e-8)
            
            # Create trial group
            trial_key = f'trial_{trial_idx:04d}'
            grp = f.create_group(trial_key)
            
            # Save input features
            grp.create_dataset('input_features', data=input_features.astype(np.float32))
            
            # Save transcription as ASCII codes (compatible with existing format)
            sentence = sentences[trial_idx].strip() if trial_idx < len(sentences) else ''
            ascii_codes = np.array([ord(c) for c in sentence] + [0] * (500 - len(sentence)), dtype=np.int32)
            grp.create_dataset('transcription', data=ascii_codes[:500])
            
            # Note: No phoneme GT (seq_class_ids) available in MAT files
        
        # Add metadata
        f.attrs['source'] = 'brain2voice_mat'
        f.attrs['original_file'] = os.path.basename(mat_path)
        f.attrs['participant'] = str(mat_data.get('participant', ['T15'])[0])
        f.attrs['session'] = str(mat_data.get('session', ['unknown'])[0])
    
    print(f"✓ Saved to {output_path}")
    return output_path


def convert_folder(folder_path, output_folder=None):
    """Convert all MAT files in a folder to HDF5."""
    mat_files = glob.glob(os.path.join(folder_path, '*.mat'))
    
    if output_folder is None:
        output_folder = folder_path
    
    os.makedirs(output_folder, exist_ok=True)
    
    converted = []
    for mat_path in mat_files:
        output_path = os.path.join(output_folder, Path(mat_path).stem + '.hdf5')
        try:
            preprocess_mat_to_hdf5(mat_path, output_path)
            converted.append(output_path)
        except Exception as e:
            print(f"✗ Failed to convert {mat_path}: {e}")
    
    return converted


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python convert_mat_to_hdf5.py <file.mat>          # Convert single file")
        print("  python convert_mat_to_hdf5.py <folder>            # Convert all MAT files in folder")
        print("  python convert_mat_to_hdf5.py <file.mat> <out.hdf5>  # Convert with custom output")
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    if os.path.isfile(input_path):
        preprocess_mat_to_hdf5(input_path, output_path)
    elif os.path.isdir(input_path):
        convert_folder(input_path, output_path)
    else:
        print(f"Error: {input_path} not found")
        sys.exit(1)
