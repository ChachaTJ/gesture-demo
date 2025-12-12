import h5py
import numpy as np

# Check structure of one HDF5 file
filepath = "/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final/t15.2023.08.11/data_train.hdf5"

with h5py.File(filepath, 'r') as f:
    print("=== Top level keys ===")
    for key in list(f.keys())[:5]:
        print(f"  {key}")
    
    print("\n=== First trial structure ===")
    trial = f['trial_0000']
    for key in trial.keys():
        data = trial[key]
        if hasattr(data, 'shape'):
            print(f"  {key}: shape={data.shape}, dtype={data.dtype}")
        else:
            print(f"  {key}: {type(data)}")
    
    print("\n=== Input features details ===")
    features = trial['input_features'][:]
    print(f"  Shape: {features.shape}")
    print(f"  Min: {features.min():.4f}, Max: {features.max():.4f}")
    print(f"  Sample row (first 10 values): {features[0, :10]}")
