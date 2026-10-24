"""Data loading, the identity-wise split and the central occlusion mask."""
from __future__ import annotations
import numpy as np

IMG_SIDE = 64
N_CLASSES = 40


def load_olivetti(synthetic: bool = False, seed: int = 0):
    """Return (X, y) with X of shape (400, 4096) in [0, 1] and y in 0..39.

    ``synthetic=True`` builds a random 40-identity stand-in with the same shape.
    It is used by the unit tests so they run without downloading the dataset.
    """
    if synthetic:
        rng = np.random.default_rng(seed)
        centres = rng.random((N_CLASSES, IMG_SIDE * IMG_SIDE))
        y = np.repeat(np.arange(N_CLASSES), 10)
        X = np.clip(centres[y] + 0.15 * rng.standard_normal((400, IMG_SIDE * IMG_SIDE)), 0, 1)
        return X.astype(np.float32), y
    from sklearn.datasets import fetch_olivetti_faces
    d = fetch_olivetti_faces()  # shuffle=False -> images sorted by identity, 10 each
    return d.data, d.target


def make_split(y: np.ndarray, seed: int, n_train: int = 6, n_val: int = 2, n_test: int = 2):
    """Seeded within-identity permutation -> train / val / test index arrays.

    NOTE: the split actually used in the report is the saved ``data/split.npz``;
    this function documents the protocol and is not guaranteed to reproduce it
    bit-for-bit (the original permutation code is not part of the supplied files).
    """
    rng = np.random.default_rng(seed)
    tr, va, te = [], [], []
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        tr += list(idx[:n_train]); va += list(idx[n_train:n_train + n_val])
        te += list(idx[n_train + n_val:n_train + n_val + n_test])
    return np.array(tr), np.array(va), np.array(te)


def load_split(path: str = "data/split.npz"):
    s = np.load(path)
    return s["train"], s["val"], s["test"]


def mask_images(X: np.ndarray, r0: int = 24, r1: int = 40, c0: int = 24, c1: int = 40) -> np.ndarray:
    """Zero rows r0..r1-1 and columns c0..c1-1 (16x16 = 256 px = 6.25% of the image)."""
    Xm = X.copy().reshape(-1, IMG_SIDE, IMG_SIDE)
    Xm[:, r0:r1, c0:c1] = 0.0
    return Xm.reshape(len(X), -1)
