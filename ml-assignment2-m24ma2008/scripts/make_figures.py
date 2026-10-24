#!/usr/bin/env python
"""Regenerate report Figures 5, 6 and 7 from the saved 30-refit results
(results/q3_results.pkl.gz). No dataset download or training needed.

    python scripts/make_figures.py            # writes figures/fig5_*.png, fig6_*.png, fig7_*.png
"""
import gzip, os, pickle, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from facebag import bias_variance

ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)
res = pickle.load(gzip.open(os.path.join(ROOT, "results", "q3_results.pkl.gz"), "rb"))["results"]
y = np.load(os.path.join(ROOT, "data", "split.npz"))["test"] // 10
P = lambda key, tag: np.stack([r["p"][key][tag] for r in res])
acc_per_refit = lambda key, tag: (P(key, tag).argmax(2) == y[None]).mean(1)

MODELS = [("Single tree (depth None)", "tree_None"), ("Bagging (B=200)", "bagfinal"),
          ("Boosting (depth 3, T=100)", "boost")]
bv = {k: {t: bias_variance(P(k, t), y) for t in ("test", "test_masked")} for _, k in MODELS}
names = [n for n, _ in MODELS]; x = np.arange(3)

# ---- Fig 5: bias^2 / variance bars + clean-accuracy boxplot --------------------------------
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].bar(x - .2, [bv[k]["test"]["bias2"] for _, k in MODELS], .4, label="Bias$^2$")
ax[0].bar(x + .2, [bv[k]["test"]["var"] for _, k in MODELS], .4, label="Variance")
ax[0].set_xticks(x, names, rotation=15); ax[0].legend(); ax[0].set_title("Measured bias$^2$ and variance (clean test)")
ax[1].boxplot([acc_per_refit(k, "test") for _, k in MODELS], tick_labels=names)
ax[1].set_title("Clean test accuracy over refits"); fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig5_bias_variance_and_accuracy.png"), dpi=130); plt.close(fig)

# ---- Fig 6: depth study --------------------------------------------------------------------
depths = [2, 6, None]; lab = ["2\n(shallow)", "6\n(intermediate)", "None\n(selected = deepest possible)"]
tv = [bias_variance(P(f"tree_{d}", "test"), y) for d in depths]
bg = [bias_variance(P(f"bag100_{d}", "test"), y) for d in depths]
gain = np.stack([acc_per_refit(f"bag100_{d}", "test") - acc_per_refit(f"tree_{d}", "test") for d in depths])
fig, ax = plt.subplots(1, 3, figsize=(15.5, 3.8)); xs = np.arange(3)
for a, key, title in ((ax[0], "bias2", "Bias$^2$ vs depth"), (ax[1], "var", "Variance vs depth")):
    a.bar(xs - .2, [t[key] for t in tv], .4, label="single tree"); a.bar(xs + .2, [b[key] for b in bg], .4, label="bagging (100)")
    a.set_xticks(xs, lab, fontsize=8); a.set_title(title); a.legend(); a.grid(alpha=.3)
ax[2].bar(xs, gain.mean(1), .5, yerr=gain.std(1, ddof=1) / np.sqrt(gain.shape[1]), capsize=4)
ax[2].set_xticks(xs, lab, fontsize=8); ax[2].set_title("Accuracy gain of bagging over single tree (± s.e.)"); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig6_bias_variance_by_depth.png"), dpi=130); plt.close(fig)

# ---- Fig 7: clean vs masked ----------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15.5, 3.8))
for off, tag, lbl in ((-.2, "test", "clean"), (.2, "test_masked", "masked")):
    ax[0].bar(x + off, [bv[k][tag]["var"] for _, k in MODELS], .4, label=lbl)
    ax[1].bar(x + off, [bv[k][tag]["acc_mean"] for _, k in MODELS], .4,
              yerr=[bv[k][tag]["acc_std"] for _, k in MODELS], capsize=3, label=lbl)
for a, t in ((ax[0], "Variance: clean vs masked test"), (ax[1], "Accuracy (mean ± std over refits)")):
    a.set_xticks(x, names, rotation=15, fontsize=8); a.set_title(t); a.legend(); a.grid(alpha=.3)
for n, k in MODELS:
    ax[2].scatter(bv[k]["test"]["var"], bv[k]["test"]["acc_mean"], s=60); ax[2].annotate(n, (bv[k]["test"]["var"], bv[k]["test"]["acc_mean"]), fontsize=8)
ax[2].set_xlabel("variance"); ax[2].set_ylabel("clean test accuracy"); ax[2].set_title("Variance vs accuracy"); ax[2].grid(alpha=.3)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_variance_clean_vs_masked.png"), dpi=130); plt.close(fig)
print("wrote fig5, fig6, fig7 to figures/")
