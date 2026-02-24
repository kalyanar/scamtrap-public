"""TF-IDF + Logistic Regression baseline."""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TfIdfLogRegBaseline:
    """TF-IDF vectorization + Logistic Regression classifier.

    For embedding-based metrics (retrieval, clustering), the TF-IDF
    vectors are used directly as the embedding representation.
    """

    def __init__(self, max_features=10000, ngram_range=(1, 3), seed=42):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            strip_accents="unicode",
            lowercase=True,
        )
        self.classifier = LogisticRegression(max_iter=1000, random_state=seed)
        self.seed = seed

    def fit(self, texts: list[str], labels: np.ndarray):
        """Fit vectorizer and classifier."""
        X = self.vectorizer.fit_transform(texts)
        self.classifier.fit(X, labels)
        return self

    def get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Return TF-IDF vectors as embeddings."""
        return self.vectorizer.transform(texts).toarray()

    def predict(self, texts: list[str]) -> np.ndarray:
        """Predict labels."""
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)
