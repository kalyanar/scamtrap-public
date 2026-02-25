"""Multiple paraphrase variants for intent descriptions.

Used for sensitivity analysis: how much do Stage B results depend
on the specific wording of intent descriptions?

Each intent has 3 variants:
  - v1: Original (mechanism-focused, detailed)
  - v2: Short (concise, keyword-heavy)
  - v3: Natural (conversational, explanatory)
"""

DESCRIPTION_VARIANTS = {
    "ham": {
        "v1": (
            "This is a legitimate, non-malicious message such as a personal "
            "conversation, notification, or normal business communication."
        ),
        "v2": "A normal, legitimate message that is not spam or a scam.",
        "v3": (
            "This message is a genuine communication from a real person or "
            "service, with no deceptive or fraudulent intent."
        ),
    },
    "credential_theft": {
        "v1": (
            "This message impersonates a legitimate service to steal login "
            "credentials, passwords, or personal identity information by asking "
            "the recipient to verify or update their account details."
        ),
        "v2": (
            "Phishing message that tries to steal passwords, usernames, or "
            "login credentials through fake account verification."
        ),
        "v3": (
            "Someone is pretending to be a trusted company and asking you to "
            "enter your password or personal details on a fake website."
        ),
    },
    "delivery": {
        "v1": (
            "This message impersonates a delivery service like UPS, FedEx, or "
            "USPS, falsely claiming a package delivery issue to trick the "
            "recipient into clicking a link or providing personal information."
        ),
        "v2": (
            "Fake package delivery notification claiming a shipment problem "
            "to steal personal information or payment details."
        ),
        "v3": (
            "This pretends to be from a shipping company saying there is a "
            "problem with your package, but it is actually a scam."
        ),
    },
    "bank_alert": {
        "v1": (
            "This message impersonates a bank or financial institution, claiming "
            "suspicious activity or account problems to pressure the recipient "
            "into revealing banking credentials or making unauthorized transfers."
        ),
        "v2": (
            "Fake bank security alert about suspicious transactions or account "
            "suspension, designed to steal financial credentials."
        ),
        "v3": (
            "This message pretends to be from your bank warning about a problem "
            "with your account, but it is trying to steal your money."
        ),
    },
    "generic_scam": {
        "v1": (
            "This message is a scam or spam that uses deceptive tactics to "
            "manipulate the recipient, but does not fit a specific category. "
            "It may use urgency, false promises, or impersonation."
        ),
        "v2": (
            "A general scam message using deception, urgency, or false claims "
            "that does not match a specific fraud category."
        ),
        "v3": (
            "This is a fraudulent message trying to trick you, but it does not "
            "fall into a specific known scam pattern."
        ),
    },
    "job_offer": {
        "v1": (
            "This message offers a fraudulent employment opportunity, often "
            "promising high pay for minimal work, to collect personal information "
            "or upfront fees from the recipient."
        ),
        "v2": (
            "Fake job offer or work-from-home opportunity promising unrealistic "
            "pay to collect personal information or advance fees."
        ),
        "v3": (
            "Someone is offering you a job that sounds too good to be true, "
            "but they really want your personal information or money upfront."
        ),
    },
    "prize_lottery": {
        "v1": (
            "This message falsely claims the recipient has won a prize, lottery, "
            "or sweepstakes, and requests personal information or a fee to claim "
            "the non-existent reward."
        ),
        "v2": (
            "Fake lottery or prize notification claiming you have won something "
            "and must pay a fee or provide details to claim it."
        ),
        "v3": (
            "This message says you have won a prize or lottery, but you actually "
            "have not and they want your money or personal information."
        ),
    },
    "crypto": {
        "v1": (
            "This message promotes a fraudulent cryptocurrency investment, "
            "trading opportunity, or wallet service designed to steal funds "
            "or personal financial information from the recipient."
        ),
        "v2": (
            "Cryptocurrency or Bitcoin investment scam promising guaranteed "
            "returns or profits to steal money from investors."
        ),
        "v3": (
            "Someone is trying to get you to invest in cryptocurrency with "
            "promises of big returns, but it is a scam to take your money."
        ),
    },
    "romance": {
        "v1": (
            "This message uses emotional manipulation and fake romantic interest "
            "to build trust with the recipient, ultimately aiming to extract "
            "money or personal information through a fabricated relationship."
        ),
        "v2": (
            "Romance scam using fake emotional connection and relationship "
            "building to eventually extract money from the victim."
        ),
        "v3": (
            "This person is pretending to be romantically interested in you, "
            "but they are actually trying to trick you into sending money."
        ),
    },
}

VARIANT_NAMES = ["v1", "v2", "v3"]


def get_descriptions_for_variant(variant: str) -> dict[str, str]:
    """Get intent descriptions for a specific variant."""
    if variant not in VARIANT_NAMES:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {VARIANT_NAMES}")
    return {intent: variants[variant] for intent, variants in DESCRIPTION_VARIANTS.items()}
