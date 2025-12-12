"""
Phoneme to Sentence Conversion and WER Calculation Script
Uses Claude API to convert phoneme sequences to sentences and calculates WER.
"""

import os
import re
import csv
import json
import time
import argparse
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# pip install anthropic jiwer pandas tqdm
import pandas as pd
from jiwer import wer
import anthropic

# ============== Configuration ==============
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "INSERT_API_KEY")
MODEL_NAME = "claude-haiku-4-5-20251001"  # Fastest and cheapest option ($1/MTok input, $5/MTok output)
MAX_WORKERS = 5  # Parallel API calls (be careful with rate limits)
BATCH_SIZE = 10  # Process in batches for efficiency
SAVE_INTERVAL = 100  # Save intermediate results every N rows

# ============== Helper Functions ==============

def parse_sentence_gt(sentence_gt_str: str) -> str:
    """
    Parse the sentence_gt field which contains ASCII codes in numpy array format.
    Example: "[ 84 104 101 121  32 114 ...]" -> "They ..."
    """
    # Extract numbers from the string
    numbers = re.findall(r'\d+', sentence_gt_str)
    # Convert to characters, ignoring 0 (padding)
    chars = [chr(int(n)) for n in numbers if int(n) > 0]
    return ''.join(chars).strip()


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation except apostrophes
    text = re.sub(r"[^\w\s']", '', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


class PhonemeToSentenceConverter:
    """Converts phoneme sequences to sentences using Claude API."""
    
    def __init__(self, api_key: str, model: str = "claude-haiku-4-5-20251001"):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
    def convert_single(self, phonemes: str) -> str:
        """Convert a single phoneme sequence to a sentence."""
        prompt = f"""Convert the following ARPABET phoneme sequence to an English sentence. 
The phonemes are space-separated. 'SIL' represents silence/pause between words.

Phoneme sequence: {phonemes}

Rules:
1. Output ONLY the English sentence, nothing else.
2. Use proper capitalization and punctuation.
3. Make the sentence grammatically correct and natural.
4. If unsure about a word, use the most likely interpretation.

English sentence:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are an expert at converting ARPABET phoneme sequences to English text. ARPABET is a phonetic transcription system used in speech recognition."
            )
            return response.content[0].text.strip()
        except Exception as e:
            print(f"API Error: {e}")
            return ""
    
    def convert_batch(self, phoneme_list: List[str]) -> List[str]:
        """Convert multiple phoneme sequences in a single API call."""
        # Create a numbered list for batch processing
        phoneme_items = "\n".join([f"{i+1}. {p}" for i, p in enumerate(phoneme_list)])
        
        prompt = f"""Convert each of the following ARPABET phoneme sequences to English sentences.
The phonemes are space-separated. 'SIL' represents silence/pause between words.

{phoneme_items}

Rules:
1. Output ONLY the English sentences, one per line, with the corresponding number.
2. Format: "1. sentence" on each line.
3. Use proper capitalization and punctuation.
4. Make each sentence grammatically correct and natural.

Output:"""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=500,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                system="You are an expert at converting ARPABET phoneme sequences to English text."
            )
            
            # Parse the response
            result_text = response.content[0].text.strip()
            results = []
            
            for line in result_text.split('\n'):
                # Extract sentence after the number
                match = re.match(r'\d+\.\s*(.+)', line.strip())
                if match:
                    results.append(match.group(1).strip())
            
            # Pad with empty strings if needed
            while len(results) < len(phoneme_list):
                results.append("")
                
            return results[:len(phoneme_list)]
            
        except Exception as e:
            print(f"Batch API Error: {e}")
            return [""] * len(phoneme_list)


def calculate_wer(hypothesis: str, reference: str) -> float:
    """Calculate Word Error Rate between hypothesis and reference."""
    hyp_normalized = normalize_text(hypothesis)
    ref_normalized = normalize_text(reference)
    
    if not ref_normalized:
        return 1.0 if hyp_normalized else 0.0
    
    try:
        return wer(ref_normalized, hyp_normalized)
    except Exception:
        return 1.0


def load_csv_data(filepath: str) -> pd.DataFrame:
    """Load and parse the evaluation CSV file."""
    df = pd.read_csv(filepath)
    
    # Parse sentence_gt to actual text
    print("Parsing sentence ground truth...")
    df['sentence_gt_text'] = df['sentence_gt'].apply(parse_sentence_gt)
    
    return df


def process_data(
    df: pd.DataFrame,
    converter: PhonemeToSentenceConverter,
    output_path: str,
    use_batch: bool = True,
    max_samples: int = None
):
    """Process the data and calculate WER."""
    
    if max_samples:
        df = df.head(max_samples)
    
    total_rows = len(df)
    print(f"Processing {total_rows} samples...")
    
    # Initialize result columns
    df['predicted_sentence'] = ""
    df['wer'] = 0.0
    
    if use_batch:
        # Batch processing
        for i in tqdm(range(0, total_rows, BATCH_SIZE), desc="Converting phonemes"):
            batch_end = min(i + BATCH_SIZE, total_rows)
            phoneme_batch = df.iloc[i:batch_end]['phoneme_prediction'].tolist()
            
            # Convert batch
            sentences = converter.convert_batch(phoneme_batch)
            
            # Store results
            for j, sentence in enumerate(sentences):
                idx = i + j
                if idx < total_rows:
                    df.at[df.index[idx], 'predicted_sentence'] = sentence
                    
                    # Calculate WER
                    ref_sentence = df.iloc[idx]['sentence_gt_text']
                    wer_score = calculate_wer(sentence, ref_sentence)
                    df.at[df.index[idx], 'wer'] = wer_score
            
            # Save intermediate results
            if (i + BATCH_SIZE) % SAVE_INTERVAL == 0:
                df.to_csv(output_path, index=False)
                
            # Rate limiting
            time.sleep(0.5)
    else:
        # Single processing (more accurate but slower)
        for idx in tqdm(range(total_rows), desc="Converting phonemes"):
            phoneme = df.iloc[idx]['phoneme_prediction']
            sentence = converter.convert_single(phoneme)
            
            df.at[df.index[idx], 'predicted_sentence'] = sentence
            
            # Calculate WER
            ref_sentence = df.iloc[idx]['sentence_gt_text']
            wer_score = calculate_wer(sentence, ref_sentence)
            df.at[df.index[idx], 'wer'] = wer_score
            
            # Save intermediate results
            if (idx + 1) % SAVE_INTERVAL == 0:
                df.to_csv(output_path, index=False)
            
            # Rate limiting
            time.sleep(0.2)
    
    # Final save
    df.to_csv(output_path, index=False)
    
    return df


def print_summary(df: pd.DataFrame):
    """Print summary statistics."""
    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    
    print(f"\nTotal samples: {len(df)}")
    print(f"\nPER (Phoneme Error Rate) Statistics:")
    print(f"  Mean: {100 - df['accuracy'].mean():.2f}%")
    print(f"  (Accuracy Mean: {df['accuracy'].mean():.2f}%)")
    
    print(f"\nWER (Word Error Rate) Statistics:")
    print(f"  Mean: {df['wer'].mean()*100:.2f}%")
    print(f"  Median: {df['wer'].median()*100:.2f}%")
    print(f"  Min: {df['wer'].min()*100:.2f}%")
    print(f"  Max: {df['wer'].max()*100:.2f}%")
    print(f"  Std: {df['wer'].std()*100:.2f}%")
    
    # Perfect matches
    perfect = (df['wer'] == 0).sum()
    print(f"\nPerfect matches (WER=0): {perfect} ({perfect/len(df)*100:.2f}%)")
    
    print("\n" + "="*50)
    
    # Show some examples
    print("\nSample Results (first 5):")
    print("-"*50)
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        print(f"\n[{idx+1}] Phonemes: {row['phoneme_prediction'][:50]}...")
        print(f"    Predicted: {row['predicted_sentence']}")
        print(f"    Reference: {row['sentence_gt_text']}")
        print(f"    PER: {100-row['accuracy']:.2f}% | WER: {row['wer']*100:.2f}%")


def main():
    parser = argparse.ArgumentParser(description="Convert phonemes to sentences and calculate WER")
    parser.add_argument("--input", "-i", type=str, 
                        default="evaluation_results_greedy.csv",
                        help="Input CSV file path")
    parser.add_argument("--output", "-o", type=str,
                        default="evaluation_results_with_wer.csv",
                        help="Output CSV file path")
    parser.add_argument("--api-key", type=str,
                        default=None,
                        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)")
    parser.add_argument("--model", type=str,
                        default="claude-haiku-4-5-20251001",
                        help="Claude model to use")
    parser.add_argument("--max-samples", "-n", type=int,
                        default=None,
                        help="Maximum number of samples to process (for testing)")
    parser.add_argument("--no-batch", action="store_true",
                        help="Disable batch processing (more accurate but slower)")
    
    args = parser.parse_args()
    
    # Set API key
    api_key = args.api_key or ANTHROPIC_API_KEY
    if api_key == "your-api-key-here":
        print("Error: Please set ANTHROPIC_API_KEY environment variable or use --api-key")
        print("Example: export ANTHROPIC_API_KEY='sk-ant-...'")
        return
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_csv_data(args.input)
    print(f"Loaded {len(df)} samples")
    
    # Initialize converter
    converter = PhonemeToSentenceConverter(api_key, args.model)
    
    # Process data
    df = process_data(
        df,
        converter,
        args.output,
        use_batch=not args.no_batch,
        max_samples=args.max_samples
    )
    
    # Print summary
    print_summary(df)
    
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
