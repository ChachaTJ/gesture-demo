import sys
import os
print("Starting batch_processor script...", flush=True)

import glob
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import google.generativeai as genai
import time

# Add current directory to path to import CPUPhonemeDecoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_decoder import CPUPhonemeDecoder

# --- Configuration ---
DATA_ROOT = "/Users/chayoonmin/Downloads/nejm-brain-to-text-main/data/hdf5_data_final"
OUTPUT_CSV = "outputs/predictions_dataset.csv"
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

print(f"Configuration loaded. Data Root: {DATA_ROOT}", flush=True)

def decode_ascii_transcription(ascii_codes):
    """Convert numpy array of ASCII codes to string."""
    try:
        valid_codes = ascii_codes[ascii_codes != 0]
        chars = [chr(int(c)) for c in valid_codes]
        return "".join(chars).strip()
    except Exception as e:
        return f"[Error decoding transcription: {e}]"

def phonemes_to_sentence_llm(phoneme_string):
    if not GOOGLE_API_KEY:
        return "[MISSING_API_KEY]"
    
    try:
        time.sleep(4) # Respect free tier rate limits (15 RPM)
        genai.configure(api_key=GOOGLE_API_KEY)
        model = genai.GenerativeModel('gemini-2.0-flash')
        prompt = f"Correct this phoneme sequence to a sentence: {phoneme_string}"
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"[LLM Error: {e}]"

def process_all_files():
    print("Initializing decoder...", flush=True)
    try:
        decoder = CPUPhonemeDecoder()
    except Exception as e:
        print(f"Failed to initialize decoder: {e}", flush=True)
        return

    search_pattern = os.path.join(DATA_ROOT, "**", "*.hdf5")
    print(f"Searching for files in: {search_pattern}", flush=True)
    all_files = glob.glob(search_pattern, recursive=True)
    
    # Filter for relevant files
    files = [f for f in all_files if any(x in f for x in ['data_train.hdf5', 'data_val.hdf5', 'data_test.hdf5'])]
    
    print(f"Found {len(files)} files to process.", flush=True)
    
    results_data = []
    
    for filepath in tqdm(files, desc="Processing files"):
        path_obj = Path(filepath)
        session_name = path_obj.parent.name
        dataset_type = path_obj.stem.replace("data_", "")
        
        try:
            with h5py.File(filepath, 'r') as f:
                trial_keys = sorted([k for k in f.keys() if k.startswith('trial_')])
                
                # OPTIMIZATION: Process only first 3 trials per file
                for trial_key in trial_keys[:3]: 
                    trial_grp = f[trial_key]
                    
                    if 'input_features' not in trial_grp:
                        continue
                        
                    neural_data = trial_grp['input_features'][:]
                    if len(neural_data.shape) == 2:
                        if neural_data.shape[1] == 512:
                            neural_data = neural_data[np.newaxis, ...]
                        elif neural_data.shape[0] == 512:
                            neural_data = neural_data.T[np.newaxis, ...]
                            
                    ground_truth = ""
                    if 'transcription' in trial_grp:
                        ground_truth = decode_ascii_transcription(trial_grp['transcription'][()])
                    
                    decoded_res = decoder.decode(neural_data)
                    phoneme_str = decoded_res[0]['phoneme_string']
                    
                    print(f"  Calling LLM for: {phoneme_str[:20]}...", flush=True)
                    llm_sent = phonemes_to_sentence_llm(phoneme_str)
                    print(f"  > Sentence: {llm_sent}", flush=True)
                    
                    results_data.append({
                        'session': session_name,
                        'dataset': dataset_type,
                        'trial_id': trial_key,
                        'ground_truth': ground_truth,
                        'predicted_phonemes': phoneme_str,
                        'llm_corrected_sentence': llm_sent
                    })

            # Incremental Save (moved outside 'with h5py' but inside 'try')
            if len(results_data) > 0 and len(results_data) % 5 == 0:
                df_temp = pd.DataFrame(results_data)
                os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
                df_temp.to_csv(OUTPUT_CSV, index=False)
                print(f"  [Saved progress to {OUTPUT_CSV}]", flush=True)

        except Exception as e:
            print(f"Error processing {filepath}: {e}", flush=True)
            continue

    df = pd.DataFrame(results_data)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved results to {OUTPUT_CSV}", flush=True)

if __name__ == "__main__":
    process_all_files()
