# Decision Trees, Bagging and SAMME Boosting on Olivetti Faces

Accuracy, stability and robustness to missing pixels — Assignment 2, roll no. **M24MA2008**, seed **242008**.

Compares a decision-tree baseline, bootstrap aggregation (bagging) and a from-scratch multiclass
**SAMME** boosting implementation on the Olivetti Faces dataset (40 identities, 64×64 grayscale,
raw 4096-pixel features). Models are evaluated on a clean test set and on a copy with the central
16×16 region zeroed out (6.25% of the pixels), and bias²/variance are measured empirically over
30 identity-wise bootstrap refits.

📄 Full write-up: [`report/report.pdf`](report/report.pdf)

## Headline results

| Model | Clean test acc. | Masked test acc. | Bias² | Variance | Clean acc. over 30 refits |
|---|---|---|---|---|---|
| Single tree (unlimited depth, leaf 1) | 61.25% | 35.00% | 0.362 | 0.634 | 0.502 ± 0.054 |
| Bagging (200 trees) | 90.00% | 81.25% | 0.486 | 0.022 | 0.822 ± 0.032 |
| SAMME (depth 3, 100 rounds) | 41.25% | 23.75% | 0.919 | 0.015 | 0.185 ± 0.055 |

Take-aways: bagging is the most accurate *and* stable; SAMME has the lowest variance only because its
output is nearly flat (low variance ≠ good accuracy); the single tree is the most sensitive to the
missing region. The test set has only 80 images (1 image = 1.25 points), so small differences should
not be over-interpreted. See the report for the caveats (e.g. the selected SAMME setting lies on the
edge of the required grid).

## Protocol

* **Split:** per identity 6 train / 2 validation / 2 test images → 240 / 80 / 80, fixed in [`data/split.npz`](data/split.npz) (indices into the sklearn Olivetti array, which is sorted by identity with 10 images each).
* **Selection:** validation accuracy only; ties go to the simpler / first-in-grid setting. Test set touched only for final evaluation.
* **Tree grid:** depth ∈ {2, 4, 8, 16, None} × min leaf ∈ {1, 2, 4}.
* **Bagging:** 200 bootstrap trees fit once; the first *B* trees form each ensemble; OOB accuracy/coverage, pairwise disagreement and vote margin tracked.
* **SAMME:** depth ∈ {1, 2, 3, 4}, rounds ∈ {10, 25, 50, 100}; discards a weak learner with ε ≥ 1 − 1/K and stops.
* **Bias/variance:** 30 refits on within-identity bootstrap resamples of the 6 training faces; Bias² and Var in probability-vector space, with the check MSE = Bias² + Var.

## Repository layout

```
├── README.md
├── requirements.txt
├── configs/config.json          # seed, grids, mask geometry, #refits
├── data/split.npz               # the fixed train/val/test index split
├── src/facebag/                 # the implementation
│   ├── data.py                  #   loading, split protocol, occlusion mask
│   ├── bagging.py               #   BaggedTrees (+ OOB, disagreement, margin)
│   ├── samme.py                 #   SAMME from scratch (+ weak-learner safeguard)
│   └── biasvar.py               #   identity-wise bootstrap, Bias²/Var
├── scripts/
│   ├── run_experiments.py       # run tree / bagging / samme / biasvar stages
│   └── analyze_refits.py        # recompute bias/variance tables from saved refits
├── tests/test_core.py           # unit tests (incl. the 3-class toy round of Table 3)
├── results/q3_results.pkl.gz    # saved class-probability vectors from the 30 refits
├── figures/                     # figures used in the report
└── report/                      # report.pdf + LaTeX support files (CVPR style, refs.bib)
```

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                       # unit tests (no dataset download needed)
python scripts/analyze_refits.py                # reproduce Tables 7, 8, 10 from the saved refits (seconds)

python scripts/run_experiments.py --stage all   # full re-run; downloads Olivetti via scikit-learn
python scripts/run_experiments.py --stage all --synthetic --quick   # ~1 min pipeline smoke test
```

The 30-refit stage is the expensive part (the report notes ≈219 minutes on two CPU workers).
Results are written to `results/out/*.json`.

## Reproducibility notes

* `scripts/analyze_refits.py` recomputes the report's bias/variance/accuracy numbers exactly from `results/q3_results.pkl.gz`.
* A fresh `run_experiments.py` run uses scikit-learn's tree implementation with fixed seeds, but the original per-tree random seeds are not part of this repository, so individual numbers can differ slightly from the report (the qualitative conclusions should hold).
* `data.make_split` documents the splitting protocol; use the saved `data/split.npz` for results comparable to the report.

## References

Breiman (1996) *Bagging predictors*; Freund & Schapire (1997) *A decision-theoretic generalization of on-line learning…*;
Zhu, Zou, Rosset & Hastie (2009) *Multi-class AdaBoost*; Geman, Bienenstock & Doursat (1992); Samaria & Harter (1994); Pedregosa et al. (2011). Full BibTeX in [`report/refs.bib`](report/refs.bib).
