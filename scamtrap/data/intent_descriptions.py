"""Natural language intent descriptions for CLIP-style alignment."""

# These descriptions are used as text prototypes in Stage B.
# Each describes the MECHANISM of the scam (how it works),
# not just the topic. This helps the model learn to generalize.

INTENT_DESCRIPTIONS = {
    "ham": (
        "This is a legitimate, non-malicious message such as a personal "
        "conversation, notification, or normal business communication."
    ),
    "credential_theft": (
        "This message impersonates a legitimate service to steal login "
        "credentials, passwords, or personal identity information by asking "
        "the recipient to verify or update their account details."
    ),
    "delivery": (
        "This message impersonates a delivery service like UPS, FedEx, or "
        "USPS, falsely claiming a package delivery issue to trick the "
        "recipient into clicking a link or providing personal information."
    ),
    "bank_alert": (
        "This message impersonates a bank or financial institution, claiming "
        "suspicious activity or account problems to pressure the recipient "
        "into revealing banking credentials or making unauthorized transfers."
    ),
    "generic_scam": (
        "This message is a scam or spam that uses deceptive tactics to "
        "manipulate the recipient, but does not fit a specific category. "
        "It may use urgency, false promises, or impersonation."
    ),
    "job_offer": (
        "This message offers a fraudulent employment opportunity, often "
        "promising high pay for minimal work, to collect personal information "
        "or upfront fees from the recipient."
    ),
    "prize_lottery": (
        "This message falsely claims the recipient has won a prize, lottery, "
        "or sweepstakes, and requests personal information or a fee to claim "
        "the non-existent reward."
    ),
    # --- HOLDOUT INTENTS (only used at test time for zero-shot) ---
    "crypto": (
        "This message promotes a fraudulent cryptocurrency investment, "
        "trading opportunity, or wallet service designed to steal funds "
        "or personal financial information from the recipient."
    ),
    "romance": (
        "This message uses emotional manipulation and fake romantic interest "
        "to build trust with the recipient, ultimately aiming to extract "
        "money or personal information through a fabricated relationship."
    ),
}

# Which intents are used during training (excludes holdout)
SEEN_INTENTS = [
    "ham", "credential_theft", "delivery", "bank_alert",
    "generic_scam", "job_offer", "prize_lottery",
]

HOLDOUT_INTENTS = ["crypto", "romance"]

# All scam intents (excluding ham)
ALL_SCAM_INTENTS = [
    "credential_theft", "delivery", "bank_alert", "generic_scam",
    "job_offer", "prize_lottery", "crypto", "romance",
]


def get_seen_and_holdout(holdout_list):
    """Derive seen/holdout intent lists from a holdout specification.

    Args:
        holdout_list: list of intent names to hold out

    Returns:
        (seen_intents, holdout_intents) where seen_intents includes 'ham'
    """
    holdout_set = set(holdout_list)
    seen = ["ham"] + [i for i in ALL_SCAM_INTENTS if i not in holdout_set]
    return seen, list(holdout_list)
