import os, sys
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from facebag import load_olivetti, load_split, mask_images, BaggedTrees, SAMME, bias_variance
from facebag.samme import samme_step

ROOT = os.path.join(os.path.dirname(__file__), "..")


def test_toy_round_matches_report_table3():
    """Report Table 3: 10 samples, 3 classes, samples 4,5,6 wrong -> eps=0.30, alpha=1.5404."""
    w = np.full(10, 0.1); wrong = np.zeros(10, bool); wrong[[4, 5, 6]] = True
    eps, alpha, w_new, Z = samme_step(w, wrong, K=3)
    assert eps == pytest.approx(0.30)
    assert alpha == pytest.approx(np.log(0.7 / 0.3) + np.log(2), abs=1e-4)      # 1.5404
    assert Z == pytest.approx(2.1, abs=1e-3)
    assert w_new[wrong] == pytest.approx(0.222, abs=1e-3) and w_new[~wrong] == pytest.approx(0.048, abs=1e-3)
    assert w_new[wrong].sum() == pytest.approx(1 - 1 / 3, abs=1e-6)             # misclassified mass = 1 - 1/K


def test_saved_split_is_identity_wise_and_disjoint():
    tr, va, te = load_split(os.path.join(ROOT, "data", "split.npz"))
    assert (len(tr), len(va), len(te)) == (240, 80, 80)
    assert len(set(tr) | set(va) | set(te)) == 400
    for part, n in ((tr, 6), (va, 2), (te, 2)):
        assert (np.bincount(part // 10, minlength=40) == n).all()


def test_mask_zeroes_256_pixels():
    X, _ = load_olivetti(synthetic=True)
    Xm = mask_images(X[:3] + 1.0).reshape(3, 64, 64)
    assert (Xm == 0).sum() == 3 * 256 and (Xm[:, 24:40, 24:40] == 0).all()


def test_bias_variance_decomposition_and_flat_reference():
    rng = np.random.default_rng(0); y = rng.integers(0, 40, 20)
    P = rng.dirichlet(np.ones(40), size=(7, 20))
    bv = bias_variance(P, y)                                   # asserts MSE == Bias2 + Var internally
    flat = bias_variance(np.full((5, 20, 40), 1 / 40), y)
    assert flat["bias2"] == pytest.approx(0.975) and flat["var"] == pytest.approx(0.0)


def test_bagging_and_samme_smoke_on_synthetic_data():
    X, y = load_olivetti(synthetic=True)
    tr, va, te = load_split(os.path.join(ROOT, "data", "split.npz"))
    bag = BaggedTrees(8, None, 1, seed=0).fit(X[tr], y[tr])
    assert bag.in_bag_.sum(1).mean() / 240 == pytest.approx(0.632, abs=0.05)    # ~63.2% distinct per bootstrap
    assert (bag.predict(X[te]) == y[te]).mean() > 0.5   # far above 1/40 chance
    V = bag.vote_fractions(X[te]); assert np.allclose(V.sum(1), 1)
    sm = SAMME(5, 3, seed=0).fit(X[tr], y[tr])
    q = sm.predict_proba(X[te]); assert np.allclose(q.sum(1), 1) and (sm.alphas_ > 0).all()
