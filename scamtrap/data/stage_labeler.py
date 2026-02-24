"""Scam stage labeling for conversation turns.

6-stage lifecycle:
  0 = Hook: initial contact, greeting
  1 = Trust Building: establish authority/credibility
  2 = Urgency: create time pressure
  3 = Info Request: ask for personal info
  4 = Payment Attempt: request money/gift cards
  5 = Escalation/Abandon: threats or gives up
"""

import re

STAGE_RULES = {
    0: {  # Hook
        "keywords": [
            r"\bhello\b", r"\bhi\b", r"\bcalling from\b", r"\bthis is\b",
            r"\bhow are you\b", r"\bgood (morning|afternoon|evening)\b",
            r"\bmy name is\b", r"\bspeaking with\b",
        ],
        "position_weight": lambda pos: 1.0 if pos < 0.2 else 0.3,
    },
    1: {  # Trust Building
        "keywords": [
            r"\bour records\b", r"\bfor your (safety|protection)\b",
            r"\bwe (noticed|detected|found)\b", r"\bauthorized\b",
            r"\bofficial\b", r"\bdepartment\b", r"\bgovernment\b",
            r"\bconfidential\b", r"\btrust\b", r"\bdon'?t worry\b",
        ],
        "position_weight": lambda pos: 1.0 if 0.1 < pos < 0.5 else 0.5,
    },
    2: {  # Urgency
        "keywords": [
            r"\bimmediately\b", r"\bright now\b", r"\burgent\b",
            r"\bsuspended\b", r"\bexpire\b", r"\blimited time\b",
            r"\bwithin (24|\d+) hours\b", r"\blast chance\b",
            r"\bact (now|fast|quickly)\b", r"\bdon'?t delay\b",
        ],
        "position_weight": lambda pos: 1.0 if 0.2 < pos < 0.7 else 0.4,
    },
    3: {  # Info Request
        "keywords": [
            r"\bsocial security\b", r"\bssn\b", r"\baccount number\b",
            r"\bdate of birth\b", r"\bverify your\b", r"\bneed your\b",
            r"\bprovide (your|me|us)\b", r"\bconfirm your\b",
            r"\bcredit card\b", r"\bpassword\b", r"\baddress\b",
        ],
        "position_weight": lambda pos: 1.0 if 0.3 < pos < 0.8 else 0.4,
    },
    4: {  # Payment Attempt
        "keywords": [
            r"\btransfer\b", r"\bpayment\b", r"\bgift card\b",
            r"\bwire\b", r"\bbitcoin\b", r"\bsend money\b",
            r"\bdeposit\b", r"\bprocessing fee\b", r"\bpay\b",
            r"\bgoogle play\b", r"\bitunes\b", r"\bzelle\b",
        ],
        "position_weight": lambda pos: 1.0 if pos > 0.4 else 0.3,
    },
    5: {  # Escalation/Abandon
        "keywords": [
            r"\blegal action\b", r"\barrested\b", r"\bpolice\b",
            r"\bfinal warning\b", r"\bgoodbye\b", r"\bdisconnect\b",
            r"\bwarrant\b", r"\bjail\b", r"\bconsequences\b",
            r"\bhang up\b",
        ],
        "position_weight": lambda pos: 1.0 if pos > 0.6 else 0.2,
    },
}

STAGE_NAMES = [
    "hook", "trust_building", "urgency",
    "info_request", "payment_attempt", "escalation",
]


class ScamStageLabeler:
    """Assign scam stage labels to conversation turns."""

    def label_turns(self, turns, is_scam=True, mode="hybrid"):
        """Label each turn with a scam stage (0-5).

        For non-scam conversations, all turns get stage 0.
        For scam conversations, uses keyword matching + position weighting.

        Args:
            turns: list of dicts with 'text' key
            is_scam: whether this is a scam conversation
            mode: "hybrid" (default), "keyword_only", or "position_only"
        """
        if not is_scam:
            return [0] * len(turns)

        n = len(turns)
        stages = []

        for i, turn in enumerate(turns):
            position = i / max(n - 1, 1)  # 0.0 to 1.0
            text = turn["text"].lower()

            scores = {}
            for stage_id, rule in STAGE_RULES.items():
                keyword_score = sum(
                    1 for kw in rule["keywords"]
                    if re.search(kw, text)
                )
                pos_weight = rule["position_weight"](position)

                if mode == "keyword_only":
                    # Ignore position weighting — all positions equal
                    scores[stage_id] = keyword_score
                elif mode == "position_only":
                    # Ignore keyword specificity — treat any keyword match as 1
                    scores[stage_id] = (1 if keyword_score > 0 else 0) * pos_weight
                else:  # hybrid (default)
                    scores[stage_id] = keyword_score * pos_weight

            # If no keywords matched, assign by position
            if max(scores.values()) == 0:
                if position < 0.2:
                    best = 0
                elif position < 0.4:
                    best = 1
                elif position < 0.6:
                    best = 2
                elif position < 0.8:
                    best = 3
                else:
                    best = 4
            else:
                best = max(scores, key=scores.get)

            stages.append(best)

        # Enforce soft monotonicity: smooth out decreases
        # (stages should generally increase over conversation)
        smoothed = [stages[0]]
        for i in range(1, len(stages)):
            # Don't allow more than 1-step backward
            if stages[i] < smoothed[-1] - 1:
                smoothed.append(smoothed[-1])
            else:
                smoothed.append(stages[i])

        return smoothed
