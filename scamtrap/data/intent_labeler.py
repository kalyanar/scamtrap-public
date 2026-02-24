"""Weak supervision intent labeling via keyword/regex rules."""

import re
import pandas as pd

# Intent rules: ordered by specificity (most specific first within each intent)
INTENT_RULES = {
    "credential_theft": [
        r"(?:verify|confirm|update|validate)\s+(?:your\s+)?(?:account|identity|password|credentials)",
        r"(?:ssn|social\s+security|login|password|username)",
        r"(?:sign\s+in|log\s*in)\s+(?:to|at|here)",
        r"(?:security\s+alert|unauthorized\s+access|suspicious\s+activity)",
    ],
    "delivery": [
        r"(?:package|parcel|shipment|delivery)\s+(?:tracking|status|notification|failed|held)",
        r"(?:ups|fedex|usps|dhl|royal\s+mail|amazon)\s+(?:delivery|package|tracking)",
        r"(?:tracking\s+number|delivery\s+attempt|redeliver|redelivery)",
    ],
    "bank_alert": [
        r"(?:bank|account)\s+(?:suspended|locked|limited|frozen|closed|compromised)",
        r"(?:unauthorized|suspicious)\s+(?:transaction|charge|activity|transfer)",
        r"(?:credit\s+card|debit\s+card|wire\s+transfer|bank\s+transfer)",
        r"(?:paypal|venmo|zelle|cash\s*app)\s+(?:account|payment|transfer)",
    ],
    "job_offer": [
        r"(?:hiring|job\s+offer|job\s+opportunity|employment|position\s+available)",
        r"(?:work\s+from\s+home|remote\s+(?:job|work|position))",
        r"(?:salary|compensation|income|earn)\s+(?:\$|usd|per\s+(?:hour|week|month))",
        r"(?:interview|resume|cv|apply\s+now)",
    ],
    "crypto": [
        r"(?:bitcoin|btc|ethereum|eth|crypto|blockchain|nft)",
        r"(?:crypto|bitcoin|coin)\s+(?:investment|trading|profit|mining)",
        r"(?:wallet|exchange|binance|coinbase|defi)",
    ],
    "romance": [
        r"(?:dear|my\s+(?:love|darling|sweetheart))",
        r"(?:lonely|looking\s+for\s+(?:love|partner|relationship))",
        r"(?:dating|romantic|single|marry|marriage)",
        r"(?:beautiful|handsome)\s+(?:woman|man|lady|person)",
    ],
    "prize_lottery": [
        r"(?:you(?:'ve|\s+have)?\s+won|winner|congratulations|congrats)",
        r"(?:prize|lottery|sweepstakes|jackpot|raffle)",
        r"(?:claim\s+(?:your|the)\s+(?:prize|reward|winnings))",
        r"(?:free\s+(?:gift|iphone|samsung|reward|voucher))",
    ],
}


class KeywordIntentLabeler:
    """Assign intent labels to scam messages using regex rules."""

    def __init__(self, rules: dict[str, list[str]] = None):
        self.rules = rules or INTENT_RULES
        # Compile patterns
        self.compiled = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in self.rules.items()
        }

    def label(self, text: str, binary_label: int) -> str:
        """Return intent label for a single message."""
        if binary_label == 0:
            return "ham"

        text_lower = text.lower()
        best_intent = None
        best_score = 0

        for intent, patterns in self.compiled.items():
            score = sum(1 for p in patterns if p.search(text_lower))
            if score > best_score:
                best_score = score
                best_intent = intent

        return best_intent if best_intent else "generic_scam"

    def label_with_scores(self, text: str, binary_label: int):
        """Return (best_intent, {intent: match_count}) for auditing.

        Returns:
            tuple of (intent_label, scores_dict) where scores_dict maps
            each intent to the number of keyword patterns that matched.
        """
        if binary_label == 0:
            return "ham", {}

        text_lower = text.lower()
        scores = {}
        for intent, patterns in self.compiled.items():
            scores[intent] = sum(1 for p in patterns if p.search(text_lower))

        best_score = max(scores.values()) if scores else 0
        if best_score == 0:
            return "generic_scam", scores

        best_intent = max(scores, key=scores.get)
        return best_intent, scores

    def label_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add intent_label column to DataFrame."""
        df = df.copy()
        df["intent_label"] = df.apply(
            lambda row: self.label(row["text"], row["binary_label"]),
            axis=1,
        )
        return df


def label_intents(df: pd.DataFrame, config=None) -> pd.DataFrame:
    """Apply intent labeling to dataset."""
    labeler = KeywordIntentLabeler()
    df = labeler.label_dataframe(df)

    # Print distribution
    print("\nIntent label distribution:")
    counts = df["intent_label"].value_counts()
    for intent, count in counts.items():
        print(f"  {intent}: {count}")

    return df
