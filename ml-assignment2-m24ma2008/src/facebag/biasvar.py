"""Empirical bias^2 / variance in probability-vector space (report Eq. 4)."""
from __future__ import annotations
import numpy as np


def identity_bootstrap(y_train, rng):
    """Sample, within each identity, as many faces as it has, with replacement."""
    idx = []
    for c in np.unique(y_train):
        members = np.where(y_train == c)[0]
        idx += list(rng.choice(members, size=len(members), replace=True))
    return np.array(idx)


def bias_variance(P, y, n_classes=40):
    """P: (R, N, K) class vectors from R refits on N fixed test points.

    Bias^2 = mean_i ||pbar(x_i) - e_i||^2,  Var = mean_{i,r} ||p_r(x_i) - pbar(x_i)||^2.
    MSE = Bias^2 + Var holds exactly (checked below).
    """
    E = np.eye(n_classes)[y]
    pbar = P.mean(0)
    bias2 = ((pbar - E) ** 2).sum(1).mean()
    var = ((P - pbar) ** 2).sum(2).mean()
    mse = ((P - E[None]) ** 2).sum(2).mean()
    assert np.isclose(mse, bias2 + var), "Bias^2 + Var must equal MSE"
    acc = (P.argmax(2) == y[None]).mean(1)
    return dict(bias2=bias2, var=var, total=bias2 + var,
                acc_mean=acc.mean(), acc_std=acc.std(ddof=1))
