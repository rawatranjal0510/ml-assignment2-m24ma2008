"""Multiclass AdaBoost (SAMME, Zhu et al. 2009) implemented from scratch."""
from __future__ import annotations
import numpy as np
from sklearn.tree import DecisionTreeClassifier

EPS_CLIP = 1e-10


def samme_step(w, wrong, K):
    """One SAMME weight update. Returns (eps, alpha, new_w, Z)."""
    eps = float(np.sum(w[wrong]))
    eps = max(eps, EPS_CLIP)
    alpha = np.log((1.0 - eps) / eps) + np.log(K - 1.0)
    w_new = w * np.exp(alpha * wrong)
    Z = w_new.sum()
    return eps, alpha, w_new / Z, Z


class SAMME:
    def __init__(self, n_rounds=100, max_depth=3, n_classes=40, seed=0):
        self.n_rounds, self.max_depth, self.K, self.seed = n_rounds, max_depth, n_classes, seed

    def fit(self, X, y):
        n = len(X)
        w = np.full(n, 1.0 / n)
        self.learners_, self.alphas_, self.eps_ = [], [], []
        self.max_w_, self.w_history_ = [], []
        self.stopped_ = None
        for t in range(self.n_rounds):
            self.w_history_.append(w.copy())
            h = DecisionTreeClassifier(max_depth=self.max_depth, random_state=self.seed + t)
            h.fit(X, y, sample_weight=w)
            wrong = (h.predict(X) != y)
            eps_raw = float(np.sum(w[wrong]))
            # weak-learner safeguard: not better than random K-class guessing -> discard and stop
            if eps_raw >= 1.0 - 1.0 / self.K:
                self.stopped_ = f"round {t}: eps={eps_raw:.3f} >= 1-1/K"
                break
            eps, alpha, w, _ = samme_step(w, wrong, self.K)
            self.learners_.append(h); self.alphas_.append(alpha); self.eps_.append(eps_raw)
            self.max_w_.append(w.max())
            if eps_raw == 0.0:           # perfect learner kept (eps clipped), weights would not change
                self.stopped_ = f"round {t}: perfect learner"
                break
        self.alphas_ = np.array(self.alphas_)
        return self

    def class_scores(self, X, R=None):
        """S_k(x) = sum_t alpha_t 1[h_t(x)=k] using the first R learners."""
        R = R or len(self.learners_)
        S = np.zeros((len(X), self.K))
        for h, a in zip(self.learners_[:R], self.alphas_[:R]):
            S[np.arange(len(X)), h.predict(X)] += a
        return S

    def predict_proba(self, X, R=None):
        """q_k(x) = S_k(x) / sum_t alpha_t (normalised weighted vote)."""
        R = R or len(self.learners_)
        return self.class_scores(X, R) / self.alphas_[:R].sum()

    def predict(self, X, R=None):
        return self.class_scores(X, R).argmax(1)
