"""NLI-based zero-shot classification baseline.

Uses a Natural Language Inference model (DeBERTa-v3-base fine-tuned on MNLI)
to perform zero-shot text classification by treating each intent label as
a hypothesis: "This message is about {intent_description}."

Reference: Yin et al., "Benchmarking Zero-shot Text Classification:
Datasets, Evaluation and Entailment Approach" (EMNLP 2019).
"""

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from tqdm import tqdm


# Default intent descriptions matching the ScamTrap intent taxonomy
DEFAULT_INTENT_HYPOTHESES = {
    "credential_theft": "This message attempts to steal login credentials or passwords.",
    "delivery": "This message is about a package delivery or shipment notification.",
    "bank_alert": "This message is a fake bank or financial account security alert.",
    "job_offer": "This message is a fraudulent job offer or work-from-home scam.",
    "prize_lottery": "This message claims the recipient has won a prize or lottery.",
    "crypto": "This message promotes cryptocurrency investment or trading scams.",
    "romance": "This message uses romantic or emotional manipulation tactics.",
    "generic_scam": "This message is a scam or fraud attempt.",
    "ham": "This message is a legitimate, non-fraudulent message.",
}


class NLIZeroShotBaseline:
    """Zero-shot classification via Natural Language Inference.

    For each message, evaluates P(entailment | message, hypothesis) across
    all intent hypotheses and selects the intent with highest entailment
    probability.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-base",
        intent_hypotheses: dict[str, str] = None,
        device: str = None,
        batch_size: int = 32,
    ):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()

        self.intent_hypotheses = intent_hypotheses or DEFAULT_INTENT_HYPOTHESES
        self.batch_size = batch_size

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model.to(self.device)

    def predict(
        self,
        texts: list[str],
        candidate_intents: list[str] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict intent labels via NLI entailment scoring.

        Args:
            texts: list of messages to classify
            candidate_intents: subset of intents to consider (default: all)

        Returns:
            (predicted_labels, score_matrix) where score_matrix is [N, C]
            with entailment probabilities for each intent.
        """
        if candidate_intents is None:
            candidate_intents = list(self.intent_hypotheses.keys())

        hypotheses = [self.intent_hypotheses[intent] for intent in candidate_intents]

        # Score matrix: [N_texts, N_intents]
        all_scores = np.zeros((len(texts), len(candidate_intents)))

        for intent_idx, hypothesis in enumerate(hypotheses):
            # Score all texts against this hypothesis in batches
            scores = []
            for i in tqdm(
                range(0, len(texts), self.batch_size),
                desc=f"NLI scoring: {candidate_intents[intent_idx]}",
                leave=False,
            ):
                batch_texts = texts[i : i + self.batch_size]
                # NLI format: (premise, hypothesis)
                inputs = self.tokenizer(
                    batch_texts,
                    [hypothesis] * len(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=256,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    logits = self.model(**inputs).logits
                    # cross-encoder/nli-deberta-v3-base label order:
                    # [contradiction=0, entailment=1, neutral=2]
                    probs = torch.softmax(logits, dim=-1)
                    entailment_scores = probs[:, 1].cpu().numpy()  # entailment class
                    scores.extend(entailment_scores)

            all_scores[:, intent_idx] = scores

        # Predict: highest entailment score wins
        pred_indices = np.argmax(all_scores, axis=1)
        pred_labels = np.array([candidate_intents[i] for i in pred_indices])

        return pred_labels, all_scores

    def evaluate(
        self,
        texts: list[str],
        true_labels: np.ndarray,
        candidate_intents: list[str] = None,
    ) -> dict:
        """Evaluate NLI zero-shot classification.

        Returns:
            dict with accuracy, f1_macro, f1_weighted, per_intent breakdown
        """
        from sklearn.metrics import accuracy_score, f1_score, classification_report

        pred_labels, scores = self.predict(texts, candidate_intents)

        results = {
            "accuracy": float(accuracy_score(true_labels, pred_labels)),
            "f1_macro": float(f1_score(true_labels, pred_labels, average="macro", zero_division=0)),
            "f1_weighted": float(f1_score(true_labels, pred_labels, average="weighted", zero_division=0)),
        }

        # Per-intent breakdown
        report = classification_report(
            true_labels, pred_labels, output_dict=True, zero_division=0
        )
        results["per_intent"] = report

        return results
