"""Bootstrap aggregation of decision trees with OOB and ensemble diagnostics."""
from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier


class BaggedTrees:
    """B trees, each fit on a size-n bootstrap sample of the training set.

    The first b trees form the b-tree ensemble, so one fit of B_max trees gives
    every smaller ensemble size (as in the report).
    """

    def __init__(self, n_trees=200, max_depth=None, min_samples_leaf=1, n_classes=40, seed=0):
        self.n_trees, self.max_depth, self.min_samples_leaf = n_trees, max_depth, min_samples_leaf
        self.n_classes, self.seed = n_classes, seed

    def fit(self, X, y):
        n = len(X)
        rng = np.random.default_rng(self.seed)
        self.trees_, self.in_bag_ = [], np.zeros((self.n_trees, n), dtype=bool)
        self.fit_time_ = np.zeros(self.n_trees)
        import time
        for b in range(self.n_trees):
            idx = rng.integers(0, n, n)
            self.in_bag_[b, np.unique(idx)] = True
            t = DecisionTreeClassifier(max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf,
                                       random_state=int(rng.integers(2**31 - 1)))
            t0 = time.perf_counter(); t.fit(X[idx], y[idx]); self.fit_time_[b] = time.perf_counter() - t0
            self.trees_.append(t)
        return self

    # --- predictions -------------------------------------------------------
    def tree_predictions(self, X, B=None):
        B = B or self.n_trees
        return np.stack([t.predict(X) for t in self.trees_[:B]])          # (B, n)

    def vote_fractions(self, X, B=None, preds=None):
        P = self.tree_predictions(X, B) if preds is None else preds
        V = np.zeros((P.shape[1], self.n_classes))
        for row in P:
            V[np.arange(len(row)), row] += 1
        return V / len(P)

    def predict(self, X, B=None):
        return self.vote_fractions(X, B).argmax(1)

    # --- diagnostics -------------------------------------------------------
    def oob_accuracy(self, X, y, B=None):
        """Return (OOB accuracy over covered samples, OOB coverage)."""
        B = B or self.n_trees
        votes = np.zeros((len(X), self.n_classes))
        for b in range(B):
            oob = ~self.in_bag_[b]
            if oob.any():
                p = self.trees_[b].predict(X[oob]); votes[np.where(oob)[0], p] += 1
        covered = votes.sum(1) > 0
        acc = (votes[covered].argmax(1) == y[covered]).mean() if covered.any() else np.nan
        return acc, covered.mean()

    @staticmethod
    def pairwise_disagreement(preds):
        """Mean over tree pairs of the fraction of samples on which the pair differs."""
        B = len(preds)
        if B < 2:
            return np.nan
        tot = sum((preds[i] != preds[j]).mean() for i in range(B) for j in range(i + 1, B))
        return tot / (B * (B - 1) / 2)

    @staticmethod
    def mean_margin(V):
        """Mean (top vote fraction - runner-up fraction)."""
        s = np.sort(V, axis=1)
        return (s[:, -1] - s[:, -2]).mean()
