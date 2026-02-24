"""Text augmentations simulating real scam obfuscation tactics."""

import random
import re

# Homoglyph mappings (Latin -> visually similar)
HOMOGLYPHS = {
    'a': ['@', '4'],
    'e': ['3'],
    'i': ['1', '!'],
    'o': ['0'],
    's': ['$', '5'],
    'l': ['1', '|'],
    't': ['+', '7'],
    'g': ['9'],
    'b': ['8'],
}

# Leetspeak mappings
LEET = {
    'a': '4', 'e': '3', 'i': '1', 'o': '0',
    's': '5', 't': '7', 'l': '1', 'g': '9',
}

# Simple synonym pairs for common scam words
SYNONYMS = {
    'click': ['tap', 'press', 'select'],
    'verify': ['confirm', 'validate', 'check'],
    'account': ['profile', 'membership'],
    'free': ['complimentary', 'no cost', 'gratis'],
    'win': ['earn', 'receive', 'get'],
    'money': ['cash', 'funds', 'payment'],
    'urgent': ['immediate', 'asap', 'critical'],
    'password': ['passcode', 'pin', 'credentials'],
    'bank': ['financial institution'],
    'prize': ['reward', 'gift', 'bonus'],
}


class ScamAugmenter:
    """Apply scam-style text augmentations."""

    def __init__(self, strategies=None, application_prob=0.5, seed=None):
        self.strategies = strategies or [
            'homoglyph', 'leetspeak', 'random_spacing', 'synonym_swap'
        ]
        self.application_prob = application_prob
        self.rng = random.Random(seed)

    def homoglyph_substitute(self, text: str, rate: float = 0.1) -> str:
        """Replace characters with visually similar Unicode/ASCII."""
        chars = list(text)
        for i, c in enumerate(chars):
            if c.lower() in HOMOGLYPHS and self.rng.random() < rate:
                chars[i] = self.rng.choice(HOMOGLYPHS[c.lower()])
        return ''.join(chars)

    def leetspeak(self, text: str, rate: float = 0.15) -> str:
        """Replace letters with leet equivalents."""
        chars = list(text)
        for i, c in enumerate(chars):
            if c.lower() in LEET and self.rng.random() < rate:
                chars[i] = LEET[c.lower()]
        return ''.join(chars)

    def random_spacing(self, text: str) -> str:
        """Insert extra spaces between some characters."""
        words = text.split()
        result = []
        for word in words:
            if len(word) > 3 and self.rng.random() < 0.3:
                # Insert space in middle of word
                pos = self.rng.randint(1, len(word) - 1)
                word = word[:pos] + ' ' + word[pos:]
            result.append(word)
        return ' '.join(result)

    def synonym_swap(self, text: str, n: int = 2) -> str:
        """Swap up to n words with synonyms."""
        words = text.split()
        swaps_done = 0
        for i, word in enumerate(words):
            if swaps_done >= n:
                break
            w_lower = word.lower().strip('.,!?')
            if w_lower in SYNONYMS:
                replacement = self.rng.choice(SYNONYMS[w_lower])
                # Preserve original casing roughly
                if word[0].isupper():
                    replacement = replacement.capitalize()
                words[i] = replacement
                swaps_done += 1
        return ' '.join(words)

    def apply_random(self, text: str, n_augmentations: int = 2) -> str:
        """Apply n random augmentations."""
        available = list(self.strategies)
        self.rng.shuffle(available)
        selected = available[:n_augmentations]

        for strategy in selected:
            if self.rng.random() > self.application_prob:
                continue
            if strategy == 'homoglyph':
                text = self.homoglyph_substitute(text)
            elif strategy == 'leetspeak':
                text = self.leetspeak(text)
            elif strategy == 'random_spacing':
                text = self.random_spacing(text)
            elif strategy == 'synonym_swap':
                text = self.synonym_swap(text)
        return text
