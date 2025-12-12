"""
Phoneme to Sentence Conversion and WER Calculation Script
Uses Claude Batch API for 50% cost savings
"""

import os
import re
import json
import time
import argparse
from typing import List, Dict
from tqdm import tqdm

# pip install anthropic jiwer pandas tqdm
import pandas as pd
from jiwer import wer as calculate_wer
import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

# ============== Configuration ==============
ANTHROPIC_API_KEY = "INSERT_API_KEY"
MODEL_NAME = "claude-haiku-4-5-20251001"  # $0.25/MTok input, $1.25/MTok output with Batch API (50% off)
BATCH_LIMIT = 10000  # Max requests per batch (API limit is 100,000)


# ============== Helper Functions ==============

def parse_sentence_gt(sentence_gt_str: str) -> str:
    """Parse ASCII codes array to text."""
    numbers = re.findall(r'\d+', str(sentence_gt_str))
    chars = [chr(int(n)) for n in numbers if int(n) > 0]
    return ''.join(chars).strip()


def normalize_text(text: str) -> str:
    """Normalize text for WER calculation."""
    text = text.lower()
    text = re.sub(r"[^\w\s']", '', text)
    text = ' '.join(text.split())
    return text


def calculate_wer_score(hypothesis: str, reference: str) -> float:
    """Calculate Word Error Rate."""
    hyp = normalize_text(hypothesis)
    ref = normalize_text(reference)
    if not ref:
        return 1.0 if hyp else 0.0
    try:
        return calculate_wer(ref, hyp)
    except:
        return 1.0


def create_prompt(phonemes: str) -> str:
    """Create a 2-step prompt for phoneme to sentence conversion."""
    # Split by SIL to show word boundaries clearly
    words = [p.strip() for p in phonemes.split('SIL') if p.strip()]
    numbered = '\n'.join([f"{i+1}. {w}" for i, w in enumerate(words)])
    
    return f"""Convert these ARPABET phoneme groups to a natural English sentence.

Phoneme groups (each group = one word):
{numbered}

Rules:
1. Each group becomes exactly ONE word
2. Adjacent phonemes within a group may form compound sounds (e.g., "AO L R EH D IY" = "already", not "all ready")
3. Consider what makes a grammatically natural sentence
4. Common patterns: "N OW" before a noun = "No", "N OW" alone or at start = "Now"

Output only the final sentence, nothing else."""


def load_and_prepare_data(filepath: str) -> pd.DataFrame:
    """Load CSV and parse sentence_gt."""
    df = pd.read_csv(filepath)
    print("Parsing ground truth sentences...")
    df['sentence_gt_text'] = df['sentence_gt'].apply(parse_sentence_gt)
    return df


def create_batch_requests(df: pd.DataFrame, start_idx: int, end_idx: int) -> List[Request]:
    """Create batch request objects."""
    requests = []
    for idx in range(start_idx, min(end_idx, len(df))):
        phonemes = df.iloc[idx]['phoneme_prediction']
        requests.append(
            Request(
                custom_id=f"req_{idx}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL_NAME,
                    max_tokens=100,
                    messages=[{
                        "role": "user",
                        "content": create_prompt(phonemes)
                    }],
                    system="You convert ARPABET phonemes to English. Each phoneme group (separated by SIL) = exactly one word. Phonemes within a group combine into a single word (e.g., 'AO L R EH D IY' = 'already'). Output ONLY the final natural sentence."
                )
            )
        )
    return requests


def submit_batch(client: anthropic.Anthropic, requests: List[Request]) -> str:
    """Submit a batch and return batch ID."""
    print(f"Submitting batch with {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    print(f"Batch created: {batch.id}")
    print(f"Status: {batch.processing_status}")
    return batch.id


def wait_for_batch(client: anthropic.Anthropic, batch_id: str, poll_interval: int = 30) -> None:
    """Poll until batch is complete."""
    print(f"\nWaiting for batch {batch_id} to complete...")
    
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        status = batch.processing_status
        counts = batch.request_counts
        
        print(f"  Status: {status} | Processing: {counts.processing} | "
              f"Succeeded: {counts.succeeded} | Errored: {counts.errored}")
        
        if status == "ended":
            print(f"\nBatch completed!")
            print(f"  Succeeded: {counts.succeeded}")
            print(f"  Errored: {counts.errored}")
            print(f"  Expired: {counts.expired}")
            print(f"  Canceled: {counts.canceled}")
            break
        
        time.sleep(poll_interval)


def retrieve_results(client: anthropic.Anthropic, batch_id: str) -> Dict[str, str]:
    """Retrieve batch results and return dict of custom_id -> sentence."""
    print(f"\nRetrieving results for batch {batch_id}...")
    
    results = {}
    error_count = 0
    
    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id
        
        if result.result.type == "succeeded":
            # Extract text from response
            message = result.result.message
            if message.content and len(message.content) > 0:
                text = message.content[0].text.strip()
                # Clean up any extra quotes or formatting
                text = text.strip('"\'')
                results[custom_id] = text
            else:
                results[custom_id] = ""
        else:
            error_count += 1
            results[custom_id] = ""
            if error_count <= 5:  # Only print first 5 errors
                print(f"  Error for {custom_id}: {result.result.type}")
    
    print(f"Retrieved {len(results)} results ({error_count} errors)")
    return results


def process_with_batch_api(
    df: pd.DataFrame,
    output_path: str,
    max_samples: int = None
) -> pd.DataFrame:
    """Process data using Batch API."""
    
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    
    if max_samples:
        df = df.head(max_samples).copy()
    
    total = len(df)
    print(f"\nTotal samples to process: {total}")
    
    # Initialize columns
    df['predicted_sentence'] = ""
    df['wer'] = 0.0
    
    # Process in batches
    all_results = {}
    
    for batch_start in range(0, total, BATCH_LIMIT):
        batch_end = min(batch_start + BATCH_LIMIT, total)
        print(f"\n{'='*50}")
        print(f"Processing batch {batch_start//BATCH_LIMIT + 1}: rows {batch_start} to {batch_end-1}")
        print(f"{'='*50}")
        
        # Create and submit batch
        requests = create_batch_requests(df, batch_start, batch_end)
        batch_id = submit_batch(client, requests)
        
        # Wait for completion
        wait_for_batch(client, batch_id)
        
        # Get results
        batch_results = retrieve_results(client, batch_id)
        all_results.update(batch_results)
        
        # Save intermediate results
        for custom_id, sentence in batch_results.items():
            idx = int(custom_id.split('_')[1])
            if idx < len(df):
                df.at[df.index[idx], 'predicted_sentence'] = sentence
                ref = df.iloc[idx]['sentence_gt_text']
                df.at[df.index[idx], 'wer'] = calculate_wer_score(sentence, ref)
        
        df.to_csv(output_path, index=False)
        print(f"Intermediate results saved to {output_path}")
    
    return df


def print_summary(df: pd.DataFrame):
    """Print evaluation summary."""
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    print(f"\nTotal samples: {len(df)}")
    
    # PER stats
    if 'accuracy' in df.columns:
        per_mean = 100 - df['accuracy'].mean()
        print(f"\nPER (Phoneme Error Rate):")
        print(f"  Mean: {per_mean:.2f}%")
    
    # WER stats
    print(f"\nWER (Word Error Rate):")
    print(f"  Mean:   {df['wer'].mean()*100:.2f}%")
    print(f"  Median: {df['wer'].median()*100:.2f}%")
    print(f"  Std:    {df['wer'].std()*100:.2f}%")
    print(f"  Min:    {df['wer'].min()*100:.2f}%")
    print(f"  Max:    {df['wer'].max()*100:.2f}%")
    
    # Perfect matches
    perfect = (df['wer'] == 0).sum()
    print(f"\nPerfect matches (WER=0): {perfect} ({perfect/len(df)*100:.2f}%)")
    
    # Show examples
    print("\n" + "-"*60)
    print("Sample Results:")
    print("-"*60)
    
    for idx in range(min(5, len(df))):
        row = df.iloc[idx]
        print(f"\n[{idx+1}]")
        print(f"  Phonemes:  {row['phoneme_prediction'][:70]}...")
        print(f"  Predicted: {row['predicted_sentence']}")
        print(f"  Reference: {row['sentence_gt_text']}")
        print(f"  WER: {row['wer']*100:.1f}%")


def main():
    parser = argparse.ArgumentParser(description="Phoneme to Sentence with Batch API")
    parser.add_argument("--input", "-i", type=str,
                        default="evaluation_results_greedy.csv",
                        help="Input CSV file")
    parser.add_argument("--output", "-o", type=str,
                        default="evaluation_results_with_wer.csv",
                        help="Output CSV file")
    parser.add_argument("--max-samples", "-n", type=int, default=None,
                        help="Max samples to process (for testing)")
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading {args.input}...")
    df = load_and_prepare_data(args.input)
    print(f"Loaded {len(df)} samples")
    
    # Process
    df = process_with_batch_api(df, args.output, args.max_samples)
    
    # Summary
    print_summary(df)
    
    print(f"\n✅ Results saved to {args.output}")


if __name__ == "__main__":
    main()
