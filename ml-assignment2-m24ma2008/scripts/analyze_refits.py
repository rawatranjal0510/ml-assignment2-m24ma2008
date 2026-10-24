#!/usr/bin/env python
"""Recompute bias^2 / variance / accuracy tables (report Tables 7, 8 and 10) from the
saved 30-refit probability vectors in results/q3_results.pkl.gz. No dataset or training needed."""
import gzip, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from facebag import bias_variance

ROOT = os.path.join(os.path.dirname(__file__), "..")
res = pickle.load(gzip.open(os.path.join(ROOT, "results", "q3_results.pkl.gz"), "rb"))["results"]
y = np.load(os.path.join(ROOT, "data", "split.npz"))["test"] // 10      # Olivetti: 10 consecutive images per identity
stack = lambda key, tag: np.stack([r["p"][key][tag] for r in res])      # (30, 80, 40)

print(f"{len(res)} refits, {len(y)} test images\n")
print(f"{'model':<18}{'Bias2':>7}{'Var':>7}{'sum':>7}{'acc(clean)':>16}   {'acc(masked)':>14}")
for name, key in [("Single tree", "tree_None"), ("Bagging (200)", "bagfinal"), ("SAMME (d3,100)", "boost")]:
    c, m = bias_variance(stack(key, "test"), y), bias_variance(stack(key, "test_masked"), y)
    print(f"{name:<18}{c['bias2']:7.3f}{c['var']:7.3f}{c['total']:7.3f}"
          f"{c['acc_mean']:9.3f}±{c['acc_std']:.3f}   {m['acc_mean']:7.3f}±{m['acc_std']:.3f}")

print("\nDepth study (Table 8): depth | tree Bias2 Var | bag100 Bias2 Var | acc tree / bag")
for d in (2, 6, None):
    t, b = bias_variance(stack(f"tree_{d}", "test"), y), bias_variance(stack(f"bag100_{d}", "test"), y)
    print(f"{str(d):>5} | {t['bias2']:.3f} {t['var']:.3f} | {b['bias2']:.3f} {b['var']:.3f} | "
          f"{t['acc_mean']:.3f} / {b['acc_mean']:.3f}  (gain {b['acc_mean'] - t['acc_mean']:+.3f})")
