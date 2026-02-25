"""E5-large frozen embedding baseline.

Uses intfloat/e5-large-v2 for frozen text embeddings + linear probe,
following the same evaluation protocol as the SBERT baseline.
E5 represents a modern, high-quality text embedding model that
significantly outperforms earlier models like MiniLM on most benchmarks.
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression


class E5LargeBaseline:
    """Pre-trained E5-large-v2 embeddings + linear classifier.

    No fine-tuning of encoder -- tests whether a modern, high-quality
    frozen embedding model is sufficient for scam intent detection.
    E5-large-v2 produces 1024-dim embeddings (vs SBERT MiniLM's 384-dim).
    """

    def __init__(self, model_name="intfloat/e5-large-v2", seed=42):
        self.encoder = SentenceTransformer(model_name)
        self.classifier = LogisticRegression(max_iter=1000, random_state=seed)
        self.seed = seed
        self.model_name = model_name

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Encode texts using frozen E5-large.

        E5 models expect a 'query: ' or 'passage: ' prefix for
        best performance. We use 'query: ' for classification.
        """
        # E5 models require instruction prefix
        prefixed_texts = [f"query: {t}" for t in texts]
        return self.encoder.encode(
            prefixed_texts, show_progress_bar=True, batch_size=32
        )

    def fit(self, texts: list[str], labels: np.ndarray):
        """Compute embeddings and fit linear classifier."""
        self.train_embeddings = self.get_embeddings(texts)
        self.classifier.fit(self.train_embeddings, labels)
        return self

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict labels."""
        embeddings = self.get_embeddings(texts)
        return self.classifier.predict(embeddings)
