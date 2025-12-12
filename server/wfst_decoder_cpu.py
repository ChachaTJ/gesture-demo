"""
Standalone WFST Beam Search Decoder for CPU
No train.py dependency
"""
import torch
import numpy as np
import pickle
from collections import defaultdict
import heapq


class WFSTBeamDecoderCPU:
    """
    WFST-based beam search decoder with 6-gram language model
    Standalone version for CPU deployment
    """

    def __init__(self, lm_path, beam_size=128, lm_weight=1.0, length_penalty=0.9):
        print(f"Loading language model from {lm_path}...")

        with open(lm_path, 'rb') as f:
            lm_data = pickle.load(f)

        self.ngram_counts = lm_data['ngram_counts']
        self.n = lm_data['order']  # 6 for 6-gram
        self.vocab = lm_data['vocab']
        self.vocab_size = len(self.vocab) + 1  # +1 for blank

        print(f"✓ Loaded {self.n}-gram LM")
        print(f"  Beam size: {beam_size}")
        print(f"  LM weight: {lm_weight}")
        print(f"  Length penalty: {length_penalty}")

        self.beam_size = beam_size
        self.lm_weight = lm_weight
        self.length_penalty = length_penalty
        self.blank_idx = 0

    def get_lm_score(self, context, next_token):
        """
        Get n-gram language model score
        Uses backoff for unseen n-grams
        """
        # Try from highest order to unigram
        for n in range(min(len(context) + 1, self.n), 0, -1):
            ngram = tuple(list(context[-(n-1):]) + [next_token]) if n > 1 else (next_token,)

            # Check if this n-gram order exists in the counts
            if n in self.ngram_counts:
                ngram_dict = self.ngram_counts[n]

                if ngram in ngram_dict:
                    count = ngram_dict[ngram]

                    # Get context count for probability
                    if n > 1:
                        context_ngram = ngram[:-1]
                        if (n-1) in self.ngram_counts and context_ngram in self.ngram_counts[n-1]:
                            context_count = self.ngram_counts[n-1][context_ngram]
                        else:
                            context_count = 1
                    else:
                        # For unigram, sum all unigram counts
                        context_count = sum(self.ngram_counts[1].values()) if 1 in self.ngram_counts else 1

                    # Log probability with smoothing
                    prob = (count + 1) / (context_count + self.vocab_size)
                    return np.log(prob)

        # Default for unseen tokens
        return np.log(1.0 / self.vocab_size)

    def decode(self, logits):
        """
        Beam search decoding with WFST and language model

        Args:
            logits: [B, T, vocab_size] or [T, vocab_size]

        Returns:
            List of decoded sequences (one per batch)
        """
        if len(logits.shape) == 3:
            # Batch decode
            return [self._decode_single(logits[i]) for i in range(logits.shape[0])]
        else:
            # Single sequence
            return [self._decode_single(logits)]

    def _decode_single(self, logits):
        """
        Decode a single sequence using beam search

        Args:
            logits: [T, vocab_size]

        Returns:
            List of token indices
        """
        T, V = logits.shape

        # Convert to log probabilities
        log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()

        # Beam: [(score, sequence, last_token, context)]
        # Start with blank token
        beam = [(0.0, [], self.blank_idx, tuple())]

        for t in range(T):
            candidates = []

            for score, seq, last_token, context in beam:
                # For each possible next token
                for token in range(V):
                    # CTC: Skip if same as last non-blank
                    if token == self.blank_idx:
                        # Blank token
                        new_seq = seq
                        new_context = context
                        new_last = self.blank_idx
                    elif token == last_token:
                        # Same token - CTC collapse
                        new_seq = seq
                        new_context = context
                        new_last = token
                    else:
                        # New token
                        new_seq = seq + [token]
                        new_context = tuple(list(context) + [token])[-5:]  # Keep last 5 for 6-gram
                        new_last = token

                    # Acoustic model score
                    am_score = log_probs[t, token]

                    # Language model score (only for non-blank new tokens)
                    if token != self.blank_idx and token != last_token:
                        lm_score = self.get_lm_score(context, token)
                    else:
                        lm_score = 0.0

                    # Combined score
                    new_score = score + am_score + self.lm_weight * lm_score

                    # Length penalty (encourage longer sequences)
                    if len(new_seq) > 0:
                        new_score = new_score / (len(new_seq) ** self.length_penalty)

                    candidates.append((new_score, new_seq, new_last, new_context))

            # Keep top beam_size candidates
            beam = heapq.nlargest(self.beam_size, candidates, key=lambda x: x[0])

        # Return best sequence
        best_score, best_seq, _, _ = max(beam, key=lambda x: x[0])

        return best_seq


if __name__ == '__main__':
    print("WFST Beam Decoder - Standalone CPU version")
    print("Ready for integration!")