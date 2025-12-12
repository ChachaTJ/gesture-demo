from cpu_decoder import CPUPhonemeDecoder
import sys
import os

# File path provided by user (found in directory)
filepath = "/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final/t15.2023.08.11/data_train.hdf5"

print(f"Running inference on: {filepath}")

if not os.path.exists(filepath):
    print(f"Error: File not found at {filepath}")
    sys.exit(1)

try:
    decoder = CPUPhonemeDecoder()
    print("Loading data...")
    # This calls the load_file method which handles hdf5 structure
    neural_data = decoder.load_file(filepath)
    print(f"Data loaded with shape: {neural_data.shape}")
    
    print("Decoding...")
    results = decoder.decode(neural_data)
    
    print("\n--- Inference Results ---")
    for res in results:
        print(f"Sample {res['sample_id']}: {res['phoneme_string']}")
        
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()
