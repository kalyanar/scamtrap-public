"""SBERT (frozen) + linear head baseline."""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class SBERTLinearBaseline:
    """Pre-trained SBERT embeddings + linear classifier.

    No fine-tuning of encoder — tests whether off-the-shelf
    embeddings are sufficient for scam intent detection.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2", seed=42):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, random_state=seed)
        self.seed = seed

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode texts using frozen SBERT."""
        return self.encoder.encode(texts, show_progress_bar=True, batch_size=64)

    def fit(self, texts: list[str], labels: np.ndarray):
        """Compute embeddings and fit linear classifier."""
        self.train_embeddings = self.get_embeddings(texts)
        self.classifier.fit(self.train_embeddings, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict labels."""
        embeddings = self.get_embeddings(texts)
        return self.classifier.predict(embeddings)
