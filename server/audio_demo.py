"""
Audio Demo Mode: Microphone → Phonemes → Sentence
For demonstrations without EEG equipment.

Uses Wav2Vec2 phoneme recognition model locally.
"""
import torch
import numpy as np
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
import sounddevice as sd
import queue
import sys
import json
import requests

# Configuration
SAMPLE_RATE = 16000  # Required for Wav2Vec2
CHUNK_DURATION = 3.0  # Seconds per chunk
API_URL = "http://localhost:5001"

# Phoneme mapping from IPA to ARPAbet (simplified)
IPA_TO_ARPABET = {
    'ɑ': 'AA', 'æ': 'AE', 'ʌ': 'AH', 'ɔ': 'AO', 'aʊ': 'AW', 'aɪ': 'AY',
    'b': 'B', 'tʃ': 'CH', 'd': 'D', 'ð': 'DH',
    'ɛ': 'EH', 'ɝ': 'ER', 'eɪ': 'EY',
    'f': 'F', 'ɡ': 'G', 'h': 'HH',
    'ɪ': 'IH', 'i': 'IY',
    'dʒ': 'JH', 'k': 'K', 'l': 'L', 'm': 'M', 'n': 'N', 'ŋ': 'NG',
    'oʊ': 'OW', 'ɔɪ': 'OY',
    'p': 'P', 'r': 'R', 's': 'S', 'ʃ': 'SH',
    't': 'T', 'θ': 'TH',
    'ʊ': 'UH', 'u': 'UW',
    'v': 'V', 'w': 'W', 'j': 'Y', 'z': 'Z', 'ʒ': 'ZH',
    ' ': 'SIL', '|': 'SIL'
}

class AudioPhonemeConverter:
    """Convert audio to phonemes using Wav2Vec2."""
    
    def __init__(self, model_name="facebook/wav2vec2-lv-60-espeak-cv-ft"):
        print(f"Loading phoneme recognition model: {model_name}")
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.model.eval()
        print("✓ Model loaded!")
    
    def audio_to_phonemes(self, audio_array, sample_rate=16000):
        """Convert audio array to phoneme string."""
        # Resample if needed
        if sample_rate != 16000:
            import librosa
            audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
        
        # Process
        inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
        
        with torch.no_grad():
            logits = self.model(inputs.input_values).logits
        
        # Decode
        predicted_ids = torch.argmax(logits, dim=-1)
        transcription = self.processor.batch_decode(predicted_ids)[0]
        
        # Convert IPA to ARPAbet
        arpabet_phonemes = self._ipa_to_arpabet(transcription)
        
        return arpabet_phonemes, transcription
    
    def _ipa_to_arpabet(self, ipa_string):
        """Convert IPA phonemes to ARPAbet format."""
        result = []
        i = 0
        while i < len(ipa_string):
            # Try 2-character match first (for diphthongs)
            if i + 1 < len(ipa_string):
                two_char = ipa_string[i:i+2]
                if two_char in IPA_TO_ARPABET:
                    result.append(IPA_TO_ARPABET[two_char])
                    i += 2
                    continue
            
            # Single character match
            char = ipa_string[i]
            if char in IPA_TO_ARPABET:
                result.append(IPA_TO_ARPABET[char])
            elif char.isalpha():
                # Keep unknown phonemes as-is (uppercase)
                result.append(char.upper())
            # Skip non-alpha characters
            i += 1
        
        # Deduplicate consecutive phonemes
        deduped = []
        prev = None
        for p in result:
            if p != prev:
                deduped.append(p)
            prev = p
        
        return ' '.join(deduped)


def record_audio(duration=3.0, sample_rate=16000):
    """Record audio from microphone."""
    print(f"🎤 Recording for {duration} seconds...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("✓ Recording complete!")
    return audio.flatten()


def send_to_api(phonemes):
    """Send phonemes to API for sentence generation."""
    try:
        response = requests.post(
            f"{API_URL}/generate_sentence",
            json={"phonemes": phonemes},
            timeout=30
        )
        return response.json()
    except Exception as e:
        return {"error": str(e)}


def demo_mode():
    """Run interactive demo mode."""
    print("=" * 60)
    print("🎙️  AUDIO DEMO MODE - Brain-to-Text Simulator")
    print("=" * 60)
    print("\nThis simulates the phoneme decoder using your microphone.")
    print("Speak clearly and the system will convert your speech to")
    print("phonemes, then to a sentence.\n")
    
    # Initialize
    converter = AudioPhonemeConverter()
    
    print("\nCommands:")
    print("  [Enter] - Record 3 seconds of audio")
    print("  [q]     - Quit\n")
    
    while True:
        user_input = input("Press Enter to record (or 'q' to quit): ").strip().lower()
        
        if user_input == 'q':
            print("Goodbye!")
            break
        
        # Record audio
        audio = record_audio(duration=CHUNK_DURATION)
        
        # Convert to phonemes
        print("\n🔄 Converting to phonemes...")
        arpabet, ipa = converter.audio_to_phonemes(audio)
        
        print(f"\n📊 Results:")
        print(f"   IPA: {ipa}")
        print(f"   ARPAbet: {arpabet}")
        
        # Send to API for sentence
        if arpabet:
            print("\n🤖 Generating sentence...")
            result = send_to_api(arpabet)
            
            if 'primary_sentence' in result:
                print(f"\n   ✅ Primary: \"{result['primary_sentence']}\"")
                if result.get('alternatives'):
                    print(f"   📋 Alternatives: {result['alternatives']}")
            elif 'error' in result:
                print(f"   ⚠️ API Error: {result['error']}")
        
        print("\n" + "-" * 60 + "\n")


def process_audio_file(filepath):
    """Process an audio file instead of microphone."""
    import librosa
    
    converter = AudioPhonemeConverter()
    
    print(f"Loading audio file: {filepath}")
    audio, sr = librosa.load(filepath, sr=16000)
    
    arpabet, ipa = converter.audio_to_phonemes(audio)
    
    print(f"\nIPA: {ipa}")
    print(f"ARPAbet: {arpabet}")
    
    result = send_to_api(arpabet)
    print(f"\nSentence: {result.get('primary_sentence', 'N/A')}")
    
    return arpabet, result


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Process audio file
        process_audio_file(sys.argv[1])
    else:
        # Interactive demo mode
        demo_mode()
