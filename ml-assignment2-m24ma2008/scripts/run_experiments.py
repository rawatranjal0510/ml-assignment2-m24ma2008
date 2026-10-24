#!/usr/bin/env python
"""Run the experiments of the report end to end.

    python scripts/run_experiments.py --stage all            # full run (slow: hours on 2 CPUs for the refits)
    python scripts/run_experiments.py --stage tree bagging   # selected stages only
    python scripts/run_experiments.py --stage all --synthetic --quick   # smoke test, no dataset needed

Outputs go to results/out/*.json. Stages: tree, bagging, samme, biasvar.
"""
import argparse, json, os, sys, time
import numpy as np
from sklearn.tree import DecisionTreeClassifier

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from facebag import (load_olivetti, load_split, mask_images, BaggedTrees, SAMME,
                     bias_variance, identity_bootstrap)

ROOT = os.path.join(os.path.dirname(__file__), "..")
CFG = json.load(open(os.path.join(ROOT, "configs", "config.json")))
OUT = os.path.join(ROOT, "results", "out"); os.makedirs(OUT, exist_ok=True)
acc = lambda a, b: float((a == b).mean())


def save(name, obj):
    json.dump(obj, open(os.path.join(OUT, name + ".json"), "w"), indent=2, default=float)
    print(f"[saved] results/out/{name}.json")


def stage_tree(D):
    best, rows = None, []
    for d in CFG["tree_grid"]["max_depth"]:
        for leaf in CFG["tree_grid"]["min_samples_leaf"]:
            t = DecisionTreeClassifier(max_depth=d, min_samples_leaf=leaf, random_state=CFG["seed"]).fit(D["Xtr"], D["ytr"])
            r = dict(depth=d, leaf=leaf, leaves=int(t.get_n_leaves()), train=acc(t.predict(D["Xtr"]), D["ytr"]),
                     val=acc(t.predict(D["Xva"]), D["yva"]), test=acc(t.predict(D["Xte"]), D["yte"]),
                     test_masked=acc(t.predict(D["Xte_m"]), D["yte"]))
            rows.append(r)
            if best is None or r["val"] > best["val"]:       # strict '>' : ties keep the first grid entry
                best = r
    print("selected tree:", best); save("tree_grid", dict(rows=rows, selected=best))
    return best


def stage_bagging(D, tree):
    Bs = CFG["bagging"]["B_values"]; Bmax = CFG["bagging"]["B_max"]
    bag = BaggedTrees(Bmax, tree["depth"], tree["leaf"], seed=CFG["seed"]).fit(D["Xtr"], D["ytr"])
    pv = bag.tree_predictions(D["Xva"]); rows = []
    for B in Bs:
        Vv = bag.vote_fractions(D["Xva"], B, preds=pv[:B]); oa, cov = bag.oob_accuracy(D["Xtr"], D["ytr"], B)
        rows.append(dict(B=B, train=acc(bag.predict(D["Xtr"], B), D["ytr"]), val=acc(Vv.argmax(1), D["yva"]),
                         test=acc(bag.predict(D["Xte"], B), D["yte"]), oob_acc=oa, oob_cov=cov,
                         disagreement=bag.pairwise_disagreement(pv[:B]), margin=bag.mean_margin(Vv)))
    best = max(rows, key=lambda r: (r["val"], r["B"]))  # highest validation accuracy (report picks B=200)
    Vc, Vm = bag.vote_fractions(D["Xte"]), bag.vote_fractions(D["Xte_m"])
    final = dict(B=best["B"], clean_acc=acc(Vc.argmax(1), D["yte"]), masked_acc=acc(Vm.argmax(1), D["yte"]),
                 clean_conf=float(Vc.max(1).mean()), masked_conf=float(Vm.max(1).mean()))
    print("bagging:", final); save("bagging", dict(rows=rows, final=final))


def stage_samme(D):
    cfg = CFG["samme"]; rows = []; best = None
    for d in cfg["depths"]:
        m = SAMME(cfg["max_rounds"], d, seed=CFG["seed"]).fit(D["Xtr"], D["ytr"])
        for R in cfg["rounds"]:
            if R > len(m.learners_): continue
            r = dict(depth=d, rounds=R, val=acc(m.predict(D["Xva"], R), D["yva"]),
                     train=acc(m.predict(D["Xtr"], R), D["ytr"]), test=acc(m.predict(D["Xte"], R), D["yte"]),
                     test_masked=acc(m.predict(D["Xte_m"], R), D["yte"]),
                     conf_clean=float(m.predict_proba(D["Xte"], R).max(1).mean()),
                     conf_masked=float(m.predict_proba(D["Xte_m"], R).max(1).mean()))
            rows.append(r)
            if best is None or r["val"] > best["val"]: best = r
        print(f"depth {d}: stopped={m.stopped_}, eps range [{min(m.eps_):.2f},{max(m.eps_):.2f}]")
    print("selected SAMME:", best); save("samme", dict(rows=rows, selected=best))


def stage_biasvar(D, tree, quick=False):
    cfg = CFG["biasvar"]; R = 3 if quick else cfg["n_refits"]; rng = np.random.default_rng(CFG["seed"])
    nb_final = 10 if quick else cfg["bag_size_final"]; nb_depth = 5 if quick else cfg["bag_size_depth_study"]
    K, models = 40, {}
    def tree_p(m, X): 
        p = np.zeros((len(X), K)); p[:, m.classes_] = m.predict_proba(X); return p
    P = {k: {"clean": [], "masked": []} for k in
         ["tree_sel", "bag_final", "samme"] + [f"{p}_{d}" for d in cfg["depths"] for p in ("tree", "bag100")]}
    for r in range(R):
        idx = identity_bootstrap(D["ytr"], rng); X, y = D["Xtr"][idx], D["ytr"][idx]
        fits = {"tree_sel": DecisionTreeClassifier(max_depth=tree["depth"], min_samples_leaf=tree["leaf"], random_state=r).fit(X, y)}
        for d in cfg["depths"]:
            fits[f"tree_{d}"] = DecisionTreeClassifier(max_depth=d, random_state=r).fit(X, y)
        for k, m in fits.items():
            for tag, Xt in (("clean", D["Xte"]), ("masked", D["Xte_m"])): P[k][tag].append(tree_p(m, Xt))
        for k, nb, d in [("bag_final", nb_final, tree["depth"])] + [(f"bag100_{d}", nb_depth, d) for d in cfg["depths"]]:
            b = BaggedTrees(nb, d, 1, seed=1000 + r).fit(X, y)
            for tag, Xt in (("clean", D["Xte"]), ("masked", D["Xte_m"])): P[k][tag].append(b.vote_fractions(Xt))
        sm = SAMME(10 if quick else 100, 3, seed=r).fit(X, y)
        for tag, Xt in (("clean", D["Xte"]), ("masked", D["Xte_m"])): P["samme"][tag].append(sm.predict_proba(Xt))
        print(f"refit {r + 1}/{R} done", flush=True)
    out = {k: {tag: bias_variance(np.stack(v), D["yte"]) for tag, v in d.items()} for k, d in P.items()}
    save("biasvar", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", nargs="+", default=["all"], choices=["all", "tree", "bagging", "samme", "biasvar"])
    ap.add_argument("--synthetic", action="store_true", help="use a random stand-in dataset (smoke test)")
    ap.add_argument("--quick", action="store_true", help="tiny ensembles / few refits (smoke test)")
    a = ap.parse_args(); stages = {"tree", "bagging", "samme", "biasvar"} if "all" in a.stage else set(a.stage)
    if a.quick:  # shrink every grid so the whole pipeline runs in about a minute
        CFG["bagging"].update(B_values=[1, 5, 10], B_max=10)
        CFG["samme"].update(depths=[1, 3], rounds=[5, 10], max_rounds=10)
        CFG["tree_grid"].update(max_depth=[4, None], min_samples_leaf=[1])
        CFG["biasvar"].update(depths=[2, None])
    X, y = load_olivetti(synthetic=a.synthetic)
    tr, va, te = load_split(os.path.join(ROOT, CFG["split"]["file"]))
    D = dict(Xtr=X[tr], ytr=y[tr], Xva=X[va], yva=y[va], Xte=X[te], yte=y[te], Xte_m=mask_images(X[te]))
    t0 = time.time(); tree = None
    if stages & {"tree", "bagging", "biasvar"}: tree = stage_tree(D)
    if "bagging" in stages: stage_bagging(D, tree)
    if "samme" in stages: stage_samme(D)
    if "biasvar" in stages: stage_biasvar(D, tree, a.quick)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
