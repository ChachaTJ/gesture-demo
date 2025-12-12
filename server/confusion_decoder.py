"""
Confusion-Aware Phoneme Decoder
Implements 3-way scoring: Acoustic (CTC) + LM (N-gram) + Confusion (pair boosting)
"""
import json
import numpy as np
from collections import defaultdict

# Load confusion matrix
CONFUSION_MATRIX_PATH = '/Users/chayoonmin/Downloads/CPU _ Phoneme Decoder/outputs/confusion_matrix.json'

# Phoneme similarity groups (based on confusion analysis)
VOICED_UNVOICED_PAIRS = {
    ('D', 'T'), ('T', 'D'),
    ('B', 'P'), ('P', 'B'),
    ('G', 'K'), ('K', 'G'),
    ('V', 'F'), ('F', 'V'),
    ('Z', 'S'), ('S', 'Z'),
    ('JH', 'CH'), ('CH', 'JH'),
}

VOWEL_CONFUSION_PAIRS = {
    ('OY', 'AY'), ('AY', 'OY'),
    ('OW', 'UW'), ('UW', 'OW'),
    ('EH', 'IH'), ('IH', 'EH'),
    ('EH', 'AH'), ('AH', 'EH'),
    ('IH', 'IY'), ('IY', 'IH'),
    ('AE', 'EH'), ('EH', 'AE'),
    ('AA', 'AO'), ('AO', 'AA'),
}


class ConfusionAwareDecoder:
    """
    Post-processing decoder that uses confusion patterns to correct predictions.
    """
    
    def __init__(self, confusion_matrix_path=CONFUSION_MATRIX_PATH):
        self.confusion_probs = {}
        self.load_confusion_matrix(confusion_matrix_path)
    
    def load_confusion_matrix(self, path):
        """Load confusion matrix and convert to probability scores."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            
            self.top_confused_pairs = data['top_confused_pairs']
            self.total_per_phoneme = data['total_samples_per_phoneme']
            self.confusion_matrix = data['confusion_matrix']
            
            # Build confusion probability table
            # P(pred | gt) = count / total_gt
            for gt_phoneme, predictions in self.confusion_matrix.items():
                total = self.total_per_phoneme.get(gt_phoneme, 1)
                for pred_phoneme, count in predictions.items():
                    if count > 0:
                        self.confusion_probs[(gt_phoneme, pred_phoneme)] = count / total
            
            print(f"✓ Loaded confusion matrix with {len(self.confusion_probs)} pairs")
            
        except Exception as e:
            print(f"⚠ Failed to load confusion matrix: {e}")
            self.confusion_probs = {}
    
    def get_confusion_score(self, phoneme1, phoneme2):
        """
        Get confusion likelihood between two phonemes.
        Higher score = more likely to be confused.
        """
        if phoneme1 == phoneme2:
            return 1.0  # Same phoneme
        
        # Check if this is a known confusion pair
        score = self.confusion_probs.get((phoneme1, phoneme2), 0)
        
        # Boost for voiced/unvoiced pairs
        if (phoneme1, phoneme2) in VOICED_UNVOICED_PAIRS:
            score = max(score, 0.15)  # At least 15% confusion likelihood
        
        # Boost for vowel confusion pairs
        if (phoneme1, phoneme2) in VOWEL_CONFUSION_PAIRS:
            score = max(score, 0.10)  # At least 10% confusion likelihood
        
        return score
    
    def get_alternative_phonemes(self, phoneme, threshold=0.02):
        """
        Get phonemes that are commonly confused with the given phoneme.
        Returns list of (alternative_phoneme, confusion_probability).
        """
        alternatives = []
        
        for (gt, pred), prob in self.confusion_probs.items():
            # If gt is confused as this phoneme, the actual might be gt
            if pred == phoneme and prob >= threshold:
                alternatives.append((gt, prob))
        
        # Add voiced/unvoiced counterpart
        for p1, p2 in VOICED_UNVOICED_PAIRS:
            if p1 == phoneme:
                if (phoneme, p2) not in [(a[0], phoneme) for a in alternatives]:
                    alternatives.append((p2, 0.15))
                break
        
        return sorted(alternatives, key=lambda x: x[1], reverse=True)
    
    def rescore_with_context(self, phoneme_sequence, context_window=2):
        """
        Rescore phoneme sequence using confusion-aware heuristics.
        
        Args:
            phoneme_sequence: List of predicted phonemes
            context_window: Number of phones to look at for context
            
        Returns:
            Corrected phoneme sequence with confidence scores
        """
        result = []
        
        for i, phoneme in enumerate(phoneme_sequence):
            if phoneme == 'SIL':
                result.append({'phoneme': phoneme, 'confidence': 1.0, 'alternatives': []})
                continue
            
            # Get alternatives
            alternatives = self.get_alternative_phonemes(phoneme)
            
            # Calculate confidence based on how often this phoneme is correctly predicted
            correct_rate = self.confusion_probs.get((phoneme, phoneme), 0.9)
            
            result.append({
                'phoneme': phoneme,
                'confidence': correct_rate,
                'alternatives': alternatives[:3]  # Top 3 alternatives
            })
        
        return result
    
    def apply_voiced_unvoiced_correction(self, phoneme_sequence, lm_scores=None):
        """
        Apply corrections for voiced/unvoiced confusions based on context.
        """
        corrected = list(phoneme_sequence)
        corrections_made = []
        
        for i, phoneme in enumerate(phoneme_sequence):
            # Check if there's a voiced/unvoiced counterpart
            counterpart = None
            for p1, p2 in VOICED_UNVOICED_PAIRS:
                if p1 == phoneme:
                    counterpart = p2
                    break
            
            if counterpart is None:
                continue
            
            # If LM scores available, check if counterpart scores better
            if lm_scores and i < len(lm_scores):
                if lm_scores.get(counterpart, 0) > lm_scores.get(phoneme, 0) * 1.2:
                    corrected[i] = counterpart
                    corrections_made.append({
                        'position': i,
                        'original': phoneme,
                        'corrected': counterpart,
                        'reason': 'lm_score'
                    })
        
        return corrected, corrections_made
    
    def decode_with_confusion_awareness(self, phoneme_string, verbose=False):
        """
        Main decoding function with confusion awareness.
        
        Args:
            phoneme_string: Space-separated phoneme string from CTC decoder
            verbose: Print detailed analysis
            
        Returns:
            dict with original, rescored, and analysis
        """
        phonemes = phoneme_string.split()
        
        # Get rescoring with alternatives
        rescored = self.rescore_with_context(phonemes)
        
        # Calculate overall confidence
        confidences = [r['confidence'] for r in rescored if r['phoneme'] != 'SIL']
        avg_confidence = np.mean(confidences) if confidences else 1.0
        
        # Identify low-confidence positions
        low_confidence_positions = [
            i for i, r in enumerate(rescored) 
            if r['confidence'] < 0.9 and r['phoneme'] != 'SIL'
        ]
        
        # Generate analysis
        analysis = {
            'original': phoneme_string,
            'avg_confidence': float(avg_confidence),
            'low_confidence_count': len(low_confidence_positions),
            'potential_corrections': []
        }
        
        for pos in low_confidence_positions:
            r = rescored[pos]
            if r['alternatives']:
                analysis['potential_corrections'].append({
                    'position': pos,
                    'current': r['phoneme'],
                    'alternatives': r['alternatives'],
                    'confidence': r['confidence']
                })
        
        if verbose:
            print(f"\n=== Confusion-Aware Analysis ===")
            print(f"Input: {phoneme_string}")
            print(f"Average confidence: {avg_confidence:.2%}")
            print(f"Low-confidence phonemes: {len(low_confidence_positions)}")
            
            for correction in analysis['potential_corrections']:
                print(f"  Position {correction['position']}: "
                      f"{correction['current']} (conf={correction['confidence']:.2%}) → "
                      f"alternatives: {[a[0] for a in correction['alternatives'][:3]]}")
        
        return analysis
    
    def generate_alternative_sequences(self, phoneme_string, max_alternatives=3):
        """
        Generate alternative phoneme sequences by swapping low-confidence phonemes.
        
        Args:
            phoneme_string: Original phoneme string
            max_alternatives: Maximum number of alternative sequences
            
        Returns:
            List of alternative phoneme strings with scores
        """
        phonemes = phoneme_string.split()
        rescored = self.rescore_with_context(phonemes)
        
        # Find low-confidence positions
        low_conf_positions = []
        for i, r in enumerate(rescored):
            if r['confidence'] < 0.9 and r['phoneme'] != 'SIL' and r['alternatives']:
                low_conf_positions.append((i, r))
        
        if not low_conf_positions:
            return []
        
        alternatives = []
        
        # Generate alternatives by swapping each low-confidence phoneme
        for pos, r in low_conf_positions[:max_alternatives]:
            for alt_phoneme, alt_prob in r['alternatives'][:2]:
                new_phonemes = phonemes.copy()
                new_phonemes[pos] = alt_phoneme
                new_string = ' '.join(new_phonemes)
                
                # Calculate score (original confidence * swap probability)
                avg_conf = np.mean([x['confidence'] for x in rescored if x['phoneme'] != 'SIL'])
                score = avg_conf * (1 - alt_prob)  # Lower alt_prob = more likely correct
                
                alternatives.append({
                    'phonemes': new_string,
                    'score': float(score),
                    'swap': f"{r['phoneme']}→{alt_phoneme} at position {pos}"
                })
        
        # Remove duplicates and sort by score
        seen = set()
        unique_alts = []
        for alt in sorted(alternatives, key=lambda x: x['score'], reverse=True):
            if alt['phonemes'] not in seen:
                seen.add(alt['phonemes'])
                unique_alts.append(alt)
        
        return unique_alts[:max_alternatives]
    
    def get_ui_response(self, phoneme_string, include_alternatives=True):
        """
        Generate UI-friendly response with all needed data.
        
        Args:
            phoneme_string: Predicted phoneme string
            include_alternatives: Whether to generate alternative sequences
            
        Returns:
            dict with all UI-needed fields
        """
        analysis = self.decode_with_confusion_awareness(phoneme_string)
        
        # Get low-confidence phonemes for highlighting
        phonemes = phoneme_string.split()
        rescored = self.rescore_with_context(phonemes)
        
        low_confidence_phonemes = []
        for i, r in enumerate(rescored):
            if r['confidence'] < 0.9 and r['phoneme'] != 'SIL':
                low_confidence_phonemes.append({
                    'position': i,
                    'phoneme': r['phoneme'],
                    'confidence': round(r['confidence'], 3),
                    'alternatives': [
                        {'phoneme': a[0], 'probability': round(a[1], 3)} 
                        for a in r['alternatives'][:3]
                    ]
                })
        
        response = {
            'phonemes': phoneme_string,
            'confidence': round(analysis['avg_confidence'], 3),
            'low_confidence_phonemes': low_confidence_phonemes,
            'low_confidence_count': len(low_confidence_phonemes),
        }
        
        # Generate alternative sequences if requested
        if include_alternatives:
            alternatives = self.generate_alternative_sequences(phoneme_string)
            response['alternative_phoneme_sequences'] = alternatives
        
        return response


# Create confusion pair lookup for quick access
def get_confusion_pairs_summary():
    """Get summary of most confused phoneme pairs for reference."""
    pairs = {
        'voiced_unvoiced': [
            ('D', 'T', 'stop consonants'),
            ('B', 'P', 'bilabial stops'),
            ('G', 'K', 'velar stops'),
            ('V', 'F', 'labiodental fricatives'),
            ('Z', 'S', 'alveolar fricatives'),
            ('JH', 'CH', 'affricates'),
        ],
        'vowels': [
            ('IH', 'IY', 'high front vowels'),
            ('EH', 'IH', 'front vowels'),
            ('EH', 'AH', 'mid vowels'),
            ('OW', 'UW', 'back vowels'),
            ('AY', 'OY', 'diphthongs'),
        ],
        'consonants': [
            ('ER', 'R', 'r-colored vowel vs r'),
            ('N', 'D', 'alveolar consonants'),
            ('N', 'NG', 'nasal consonants'),
        ]
    }
    return pairs


if __name__ == '__main__':
    # Test the confusion-aware decoder
    decoder = ConfusionAwareDecoder()
    
    # Test cases
    test_sequences = [
        "AA R SIL DH AH SIL R EY T SIL",  # "Are they right?"
        "AY SIL L IY K SIL DH AE T SIL",  # "I like that"
        "T EH L SIL M AY SIL F AE M AH L IY SIL",  # "Tell my family"
    ]
    
    print("=" * 80)
    print("CONFUSION-AWARE DECODER TEST")
    print("=" * 80)
    
    # Show confusion pairs summary
    pairs = get_confusion_pairs_summary()
    print("\nKnown Confusion Patterns:")
    for category, pair_list in pairs.items():
        print(f"\n  {category.replace('_', ' ').title()}:")
        for p1, p2, desc in pair_list:
            score = decoder.get_confusion_score(p1, p2)
            print(f"    {p1} ↔ {p2} ({desc}): {score:.1%}")
    
    # Test sequences
    print("\n" + "=" * 80)
    print("SEQUENCE ANALYSIS")
    print("=" * 80)
    
    for seq in test_sequences:
        analysis = decoder.decode_with_confusion_awareness(seq, verbose=True)
        print()
