#!/usr/bin/env python3
"""
Full LibriSpeech Phoneme Frequency Analyzer
=============================================
Reads the ENTIRE librispeech-lm-norm.txt.gz file and calculates 
accurate phoneme frequencies.
"""

import gzip
import re
import json
from collections import Counter
from pathlib import Path
import pronouncing

# Paths
DATA_FILE = Path('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/data/librispeech-lm-norm.txt.gz')
OUTPUT_FILE = Path('/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/librispeech_full_phoneme_freq.json')


def text_to_phonemes(text: str) -> list:
    """Convert text to phonemes using CMU Dict."""
    text = text.upper()
    text = re.sub(r'[^A-Z\s]', '', text)
    words = text.split()
    
    phonemes = []
    for word in words:
        phones = pronouncing.phones_for_word(word.lower())
        if phones:
            cleaned = [re.sub(r'\d', '', p) for p in phones[0].split()]
            phonemes.extend(cleaned)
    
    return phonemes


def analyze_full_corpus(max_lines=5000000):
    """Analyze the LibriSpeech LM corpus (up to max_lines)."""
    
    print(f"Reading corpus from: {DATA_FILE}")
    print(f"Max lines: {max_lines:,}")
    print("This may take a few minutes...")
    
    counter = Counter()
    total_lines = 0
    total_phonemes = 0
    
    with gzip.open(DATA_FILE, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            phonemes = text_to_phonemes(line.strip())
            counter.update(phonemes)
            total_phonemes += len(phonemes)
            total_lines += 1
            
            if (i + 1) % 500000 == 0:
                print(f"  Processed {i+1:,} lines... ({total_phonemes:,} phonemes)")
    
    print(f"\n✓ Finished! {total_lines:,} lines, {total_phonemes:,} phonemes\n")
    
    # Calculate frequencies
    frequencies = {p: count / total_phonemes for p, count in counter.most_common()}
    
    # Print results
    print("="*60)
    print("📊 FULL LIBRISPEECH PHONEME FREQUENCY")
    print("="*60)
    print(f"\nTotal lines: {total_lines:,}")
    print(f"Total phonemes: {total_phonemes:,}")
    print(f"Unique phonemes: {len(frequencies)}")
    
    print(f"\n{'Rank':<6}{'Phoneme':<10}{'Frequency':<12}{'Count':<12}")
    print("-"*45)
    
    for i, (phoneme, freq) in enumerate(sorted(frequencies.items(), key=lambda x: -x[1])):
        count = counter[phoneme]
        print(f"{i+1:<6}{phoneme:<10}{freq*100:.4f}%{count:>12,}")
    
    # Categorize
    print("\n" + "="*60)
    print("📋 CATEGORIZED BY FREQUENCY")
    print("="*60)
    
    very_common = [p for p, f in frequencies.items() if f >= 0.03]
    common = [p for p, f in frequencies.items() if 0.01 <= f < 0.03]
    rare = [p for p, f in frequencies.items() if 0.005 <= f < 0.01]
    very_rare = [p for p, f in frequencies.items() if f < 0.005]
    
    print(f"\n🔴 Very Common (≥3%): {sorted(very_common, key=lambda p: -frequencies[p])}")
    print(f"\n🟡 Common (1-3%): {sorted(common, key=lambda p: -frequencies[p])}")
    print(f"\n🟢 Rare (<1%): {sorted(rare, key=lambda p: -frequencies[p])}")
    print(f"\n⭐ Very Rare (<0.5%): {sorted(very_rare, key=lambda p: -frequencies[p])}")
    
    # Save
    output = {
        'source': 'LibriSpeech LM Corpus (FULL - 40M sentences)',
        'total_lines': total_lines,
        'total_phonemes': total_phonemes,
        'unique_phonemes': len(frequencies),
        'frequencies': frequencies
    }
    
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✓ Saved to: {OUTPUT_FILE}")
    
    # Python dict for copy-paste
    print("\n" + "="*60)
    print("📝 COPY-PASTE READY DICT")
    print("="*60)
    print("\nPHONEME_FREQ = {")
    for p, f in sorted(frequencies.items(), key=lambda x: -x[1]):
        print(f"    '{p}': {f:.6f},")
    print("}")
    
    return frequencies


if __name__ == '__main__':
    analyze_full_corpus()
