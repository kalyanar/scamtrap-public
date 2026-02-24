"""Trajectory prediction baselines for Stage C comparison."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


class MarkovChainBaseline:
    """First-order Markov chain over scam stages.

    Learns transition matrix P[i,j] = P(stage_{t+1}=j | stage_t=i)
    from training trajectories. No embeddings needed.
    """

    def __init__(self, num_stages=6):
        self.num_stages = num_stages
        self.transition_matrix = None

    def fit(self, trajectories):
        """Learn transition matrix from training trajectories."""
        counts = np.zeros((self.num_stages, self.num_stages))

        for traj in trajectories:
            stages = traj["stages"]
            for i in range(len(stages) - 1):
                s_from = stages[i]
                s_to = stages[i + 1]
                if 0 <= s_from < self.num_stages and 0 <= s_to < self.num_stages:
                    counts[s_from, s_to] += 1

        # Normalize rows (add-1 smoothing)
        counts += 1
        self.transition_matrix = counts / counts.sum(axis=1, keepdims=True)
        return self

    def predict_stages(self, trajectories):
        """Predict next stage for each turn using Markov transitions."""
        all_preds = []
        all_labels = []

        for traj in trajectories:
            stages = traj["stages"]
            for i in range(len(stages) - 1):
                current = stages[i]
                if 0 <= current < self.num_stages:
                    pred = np.argmax(self.transition_matrix[current])
                    all_preds.append(pred)
                    all_labels.append(stages[i + 1])

        return np.array(all_preds), np.array(all_labels)

    def predict_escalation(self, trajectories, max_turns=30):
        """Predict escalation probability by simulating forward."""
        all_probs = []
        all_labels = []

        for traj in trajectories:
            stages = traj["stages"]
            for i in range(len(stages)):
                # Probability of reaching stage >= 4 within remaining turns
                current = stages[i]
                # Simple: sum of transition probs to stages 4, 5
                if 0 <= current < self.num_stages:
                    prob = sum(self.transition_matrix[current, s]
                               for s in range(4, self.num_stages))
                else:
                    prob = 0.0
                all_probs.append(prob)
                all_labels.append(1.0 if stages[i] >= 4 else 0.0)

        return np.array(all_probs), np.array(all_labels)

    def evaluate(self, trajectories):
        """Full evaluation."""
        # Stage prediction
        stage_preds, stage_labels = self.predict_stages(trajectories)
        stage_acc = float(accuracy_score(stage_labels, stage_preds))
        stage_f1 = float(f1_score(
            stage_labels, stage_preds, average="macro", zero_division=0,
        ))

        # Escalation prediction
        esc_probs, esc_labels = self.predict_escalation(trajectories)
        esc_preds = (esc_probs >= 0.5).astype(int)
        esc_acc = float(accuracy_score(esc_labels.astype(int), esc_preds))

        if len(set(esc_labels)) >= 2:
            esc_auroc = float(roc_auc_score(esc_labels, esc_probs))
        else:
            esc_auroc = float("nan")

        return {
            "stage_accuracy": stage_acc,
            "stage_f1_macro": stage_f1,
            "escalation_accuracy": esc_acc,
            "escalation_auroc": esc_auroc,
        }


class LogRegBaseline:
    """Logistic regression on single-turn embeddings (no history)."""

    def __init__(self, num_stages=6):
        self.num_stages = num_stages
        self.stage_clf = LogisticRegression(max_iter=1000)
        self.esc_clf = LogisticRegression(max_iter=1000)

    def fit(self, trajectories):
        """Fit classifiers on individual turn embeddings."""
        all_embs = []
        all_stages = []
        all_esc = []

        for traj in trajectories:
            embs = traj["embeddings"]
            stages = traj["stages"]
            for i in range(len(stages)):
                all_embs.append(embs[i])
                all_stages.append(stages[i])
                all_esc.append(1 if stages[i] >= 4 else 0)

        X = np.array(all_embs)
        self.stage_clf.fit(X, np.array(all_stages))
        self.esc_clf.fit(X, np.array(all_esc))
        return self

    def evaluate(self, trajectories):
        """Evaluate on test trajectories."""
        all_embs = []
        all_stages = []
        all_esc = []

        for traj in trajectories:
            embs = traj["embeddings"]
            stages = traj["stages"]
            for i in range(len(stages)):
                all_embs.append(embs[i])
                all_stages.append(stages[i])
                all_esc.append(1 if stages[i] >= 4 else 0)

        X = np.array(all_embs)
        stage_labels = np.array(all_stages)
        esc_labels = np.array(all_esc)

        # Stage prediction
        stage_preds = self.stage_clf.predict(X)
        stage_acc = float(accuracy_score(stage_labels, stage_preds))
        stage_f1 = float(f1_score(
            stage_labels, stage_preds, average="macro", zero_division=0,
        ))

        # Escalation prediction
        esc_probs = self.esc_clf.predict_proba(X)[:, 1]
        esc_preds = (esc_probs >= 0.5).astype(int)
        esc_acc = float(accuracy_score(esc_labels, esc_preds))

        if len(set(esc_labels)) >= 2:
            esc_auroc = float(roc_auc_score(esc_labels, esc_probs))
        else:
            esc_auroc = float("nan")

        return {
            "stage_accuracy": stage_acc,
            "stage_f1_macro": stage_f1,
            "escalation_accuracy": esc_acc,
            "escalation_auroc": esc_auroc,
        }
