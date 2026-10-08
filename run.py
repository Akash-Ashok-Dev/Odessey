#!/usr/bin/env python3
"""
run.py — experiment CLI for the intrusion detection hackathon.

Subcommands:
  bar      Reproduce baseline (HistGB, raw features) on group-split + official val
  fit      FE + LightGBM/HistGB ensemble, eval, threshold sweep, save artifact
  lomo     Leave-one-attack-family-out CV (unseen-family robustness)
  predict  Generate submission CSV from a saved artifact
"""
import argparse
import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

import ids
from ids import FEATS, CATS

TRAIN = "shared/data/train.csv"
VAL = "shared/data/validation.csv"
ARTIFACT = "artifacts/model.joblib"


def load_train(quick=False, dedupe=False):
    df = ids.load_labelled_with_dupkey(TRAIN, nrows=100000 if quick else None)
    n_conf = ids.label_conflicts(df)
    if dedupe:
        df, n_dup = ids.dedupe(df)
        print(f"[data] deduped: removed {n_dup:,} dups ({n_conf:,} label-conflict rows)")
    else:
        print(f"[data] raw={len(df):,} (contains {n_conf:,} label-conflict rows)")
    print(f"[data] attack rate: {df['Label'].mean():.4f} "
          f"(pos={int(df['Label'].sum()):,})")
    return df


def dup_stats(df):
    print(f"[data] attack rate after dedupe: {df['Label'].mean():.4f}")


# --------------------------------------------------------------- bar
def cmd_bar(a):
    from sklearn.ensemble import HistGradientBoostingClassifier
    tr = load_train(a.quick, dedupe=a.dedupe)

    tr_g, va_g = ids.group_split(tr, val_pct=15, seed=0)
    print(f"[split] group train={len(tr_g):,}  group val={len(va_g):,}")

    Xtr, maps = ids.design_matrix(tr_g, fe=False, is_train=True)
    Xva, _ = ids.design_matrix(va_g, maps=maps, fe=False)
    t = time.time()
    m = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, random_state=42,
        categorical_features=ids.cat_idx(Xtr)).fit(Xtr, tr_g["Label"].values)
    print(f"[train] HistGB raw group-train: {time.time() - t:.1f}s")

    p = m.predict_proba(Xva)[:, 1]
    yv = va_g["Label"].values
    ids.report(yv, p, cats=va_g["attack_cat"].values, w=a.w, th=0.5,
               title="BASELINE HistGB-raw | group-val @0.5")
    th, _ = ids.best_threshold(yv, p, w=a.w)
    ids.report(yv, p, cats=va_g["attack_cat"].values, w=a.w, th=th,
               title="BASELINE HistGB-raw | group-val cost-opt")

    # official validation (organizer-style: trained on full deduped train)
    Xf, mapsf = ids.design_matrix(tr, fe=False, is_train=True)
    m2 = HistGradientBoostingClassifier(
        max_iter=100, learning_rate=0.1, random_state=42,
        categorical_features=ids.cat_idx(Xf)).fit(Xf, tr["Label"].values)
    vo = ids.load(VAL, labelled=True, nrows=100000 if a.quick else None)
    Xo, _ = ids.design_matrix(vo, maps=mapsf, fe=False)
    po = m2.predict_proba(Xo)[:, 1]
    yo = vo["Label"].values
    ids.report(yo, po, cats=vo["attack_cat"].values, w=a.w, th=0.5,
               title="BASELINE HistGB-raw | OFFICIAL val @0.5")
    tho, _ = ids.best_threshold(yo, po, w=a.w)
    ids.report(yo, po, cats=vo["attack_cat"].values, w=a.w, th=tho,
               title="BASELINE HistGB-raw | OFFICIAL val cost-opt")
    op_table(yo, po, w=a.w, title="BASELINE OFFICIAL val")


def op_table(y, p, cats=None, w=20.0, title=""):
    """Print F1/cost across operating points + argmax-F1 and argmin-cost."""
    from sklearn.metrics import (f1_score, precision_score, recall_score)
    ths = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.08, 0.06,
           0.05, 0.04, 0.03, 0.02, 0.01]
    print(f"\n[operating points] {title} (w={w:.0f})")
    print(f"  {'th':>6} {'F1':>7} {'P':>7} {'R':>7} {'cost':>8}")
    for th in ths:
        m = ids.metrics(y, p, w=w, th=th)
        print(f"  {th:>6.2f} {m['f1']:>7.4f} {m['prec']:>7.4f} "
              f"{m['rec']:>7.4f} {m['cost']:>8,}")
    f1s_vec = _f1_curve(y, p)
    th_f1, best_f1 = f1s_vec
    th_c, mc = ids.best_threshold(y, p, w=w)
    mf1 = ids.metrics(y, p, w=w, th=th_f1)
    print(f"  F1-opt:  th={th_f1:.4f} F1={best_f1:.4f} cost={mf1['cost']:,} "
          f"P={mf1['prec']:.4f} R={mf1['rec']:.4f}")
    print(f"  COST-opt: th={th_c:.4f} F1={mc['f1']:.4f} cost={mc['cost']:,} "
          f"P={mc['prec']:.4f} R={mc['rec']:.4f}")
    return th_f1, best_f1, th_c, mc


def _f1_curve(y, p):
    """Vectorized argmax F1 over all probability thresholds."""
    ps = np.asarray(p, dtype=np.float64)
    ys = np.asarray(y, dtype=np.int64)
    order = np.argsort(ps, kind="mergesort")
    ps_s = ps[order]
    cum = np.concatenate([[0], np.cumsum(ys[order])])
    n = len(ps)
    P = int(cum[-1])
    cand = np.unique(ps_s)
    idx = np.searchsorted(ps_s, cand, side="left")
    tp = P - cum[idx]
    fp = (n - idx) - tp
    fn = P - tp
    f1 = 2.0 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    return float(cand[i]), float(f1[i])


# --------------------------------------------------------------- fit
def cmd_fit(a):
    tr = load_train(a.quick, dedupe=a.dedupe)
    tr_g, va_g = ids.group_split(tr, val_pct=15, seed=0)
    print(f"[split] group train={len(tr_g):,}  group val={len(va_g):,}")

    t = time.time()
    Xtr, maps = ids.design_matrix(tr_g, fe=not a.no_fe, is_train=True)
    Xva, _ = ids.design_matrix(va_g, maps=maps, fe=not a.no_fe)
    Xva = Xva[Xtr.columns]
    print(f"[feat ] {Xtr.shape[1]} columns, design {time.time() - t:.1f}s")

    ytr = tr_g["Label"].values
    yva = va_g["Label"].values
    lkw = {"n_estimators": a.trees, "num_leaves": a.leaves,
           "learning_rate": a.lgb_lr, "min_child_samples": a.mcs}
    hkw = {"max_iter": a.hgb_iter, "max_leaf_nodes": a.hgb_leaves,
           "learning_rate": a.hgb_lr}
    if a.spw > 0:
        lkw["scale_pos_weight"] = a.spw
        hkw["class_weight"] = "balanced"
        print(f"[train] positive class weighting: scale_pos_weight={a.spw}")
    sw = ids.family_weights(tr_g) if a.fam_bal else None
    if sw is not None:
        print(f"[train] family weights ON (mean={sw.mean():.2f}, max={sw.max():.2f})")
    t = time.time()
    lgbm, hgb = ids.train_ensemble(Xtr, ytr, seed=42, lgb_kw=lkw, hgb_kw=hkw, sw=sw)
    cb = None
    if a.cat:
        t2 = time.time()
        cb = ids.train_catboost(Xtr, ytr, seed=42, sw=sw)
        if cb is not None:
            print(f"[train] catboost: {time.time() - t2:.1f}s")
    blend = ("weights lgb=0.4 hgb=0.3 cat=0.3" if cb is not None
             else f"weights lgb=alpha={a.alpha} hgb={1 - a.alpha:.3f}")
    print(f"[train] ensemble: {time.time() - t:.1f}s  ({blend})")
    imp = sorted(zip(Xtr.columns, lgbm.feature_importances_), key=lambda z: -z[1])
    print("[lgb  ] top features: " + ", ".join(f"{c}:{v}" for c, v in imp[:12]))

    probs = ids.ensemble_proba(lgbm, hgb, Xva, alpha=a.alpha, cb=cb)
    ids.report(yva, probs, cats=va_g["attack_cat"].values, w=a.w, th=0.5,
               title="ENSEMBLE | group-val @0.5")
    if a.th is None and a.objective == "f1":
        th, _ = _f1_curve(yva, probs)
        th_title = "ENSEMBLE | group-val F1-opt"
    elif a.th is None:
        th, _ = ids.best_threshold(yva, probs, w=a.w)
        th_title = f"ENSEMBLE | group-val cost-opt (w={a.w:.0f})"
    else:
        th = a.th
        th_title = f"ENSEMBLE | group-val @forced th={th:.4f}"
    ids.report(yva, probs, cats=va_g["attack_cat"].values, w=a.w, th=th,
               title=th_title)
    print(f"[thresh] using th={th:.4f}")

    # official validation confirmation
    vo = ids.load(VAL, labelled=True, nrows=100000 if a.quick else None)
    Xo, _ = ids.design_matrix(vo, maps=maps, fe=not a.no_fe)
    Xo = Xo[Xtr.columns]
    po = ids.ensemble_proba(lgbm, hgb, Xo, alpha=a.alpha, cb=cb)
    ids.report(vo["Label"].values, po, cats=vo["attack_cat"].values, w=a.w,
               th=0.5, title="ENSEMBLE | OFFICIAL val @0.5")
    ids.report(vo["Label"].values, po, cats=vo["attack_cat"].values, w=a.w,
               th=th, title="ENSEMBLE | OFFICIAL val @chosen threshold")
    op_table(vo["Label"].values, po, w=a.w, title="OFFICIAL val")
    op_table(yva, probs, w=a.w, title="group-val")

    if a.save:
        # final artifact trained on ALL train (threshold from group-val)
        Xf, mapsf = ids.design_matrix(tr, fe=not a.no_fe, is_train=True)
        swf = ids.family_weights(tr) if a.fam_bal else None
        t = time.time()
        flgb, fhgb = ids.train_ensemble(Xf, tr["Label"].values, seed=42,
                                        lgb_kw=lkw, hgb_kw=hkw, sw=swf)
        print(f"[final] retrain on full train: {time.time() - t:.1f}s")
        fcb = None
        if a.cat:
            t2 = time.time()
            fcb = ids.train_catboost(Xf, tr["Label"].values, seed=42, sw=swf)
            if fcb is not None:
                print(f"[final] catboost on full train: {time.time() - t2:.1f}s")
        params = {"alpha": a.alpha, "trees": a.trees, "leaves": a.leaves,
                  "lgb_lr": a.lgb_lr, "mcs": a.mcs, "hgb_iter": a.hgb_iter,
                  "hgb_leaves": a.hgb_leaves, "hgb_lr": a.hgb_lr,
                  "cat": a.cat and fcb is not None, "fam_bal": a.fam_bal}
        art = ids.Artifact(mapsf, (flgb, fhgb), Xf.columns, th,
                           meta={"w": a.w, "fe": not a.no_fe,
                           "objective": a.objective,
                           "threshold_rationale": th_title,
                                 "train_rows": len(tr), **params},
                           alpha=a.alpha, cb=fcb)
        os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
        joblib.dump(art, ARTIFACT)
        blend = ("0.4/0.3/0.3" if fcb is not None else f"{a.alpha}/{1 - a.alpha:.2f}")
        print(f"[save ] artifact -> {ARTIFACT} (threshold={th:.4f}, "
              f"blend={blend}, feats={len(art.feature_cols)})")
        cmd_package(a) if getattr(a, "package", False) else None


# --------------------------------------------------------------- tune
def cmd_tune(a):
    """Grid search on group-val: model variants x blend weight, F1-opt objective."""
    tr = load_train(a.quick, dedupe=a.dedupe)
    tr_g, va_g = ids.group_split(tr, val_pct=15, seed=0)
    Xtr, maps = ids.design_matrix(tr_g, fe=not a.no_fe, is_train=True)
    Xva, _ = ids.design_matrix(va_g, maps=maps, fe=not a.no_fe)
    Xva = Xva[Xtr.columns]
    ytr = tr_g["Label"].values
    yva = va_g["Label"].values
    print(f"[tune ] train={len(tr_g):,} val={len(va_g):,} feats={Xtr.shape[1]}")

    LGB_VARIANTS = {
        "cur(400,64,.05,50)": {"n_estimators": 400, "num_leaves": 64,
                               "learning_rate": 0.05, "min_child_samples": 50},
        "big(800,255,.04,30)": {"n_estimators": 800, "num_leaves": 255,
                                "learning_rate": 0.04, "min_child_samples": 30},
        "deep(600,127,.03,50)": {"n_estimators": 600, "num_leaves": 127,
                                 "learning_rate": 0.03, "min_child_samples": 50},
        "small(400,63,.05,100)": {"n_estimators": 400, "num_leaves": 63,
                                  "learning_rate": 0.05, "min_child_samples": 100},
    }
    HGB_VARIANTS = {
        "cur(200,63,.08)": {"max_iter": 200, "max_leaf_nodes": 63,
                            "learning_rate": 0.08},
        "more(350,127,.05)": {"max_iter": 350, "max_leaf_nodes": 127,
                              "learning_rate": 0.05},
        "tight(300,63,.08,msl20)": {"max_iter": 300, "max_leaf_nodes": 63,
                                    "learning_rate": 0.08, "min_samples_leaf": 20},
    }

    def score(p_lgb, p_hgb):
        """Best (alpha, th, F1) on group-val; also PR-AUC at best alpha."""
        best = None
        for al in np.arange(0.0, 1.001, 0.1):
            p = al * p_lgb + (1 - al) * p_hgb
            th, m = _f1_curve(yva, p)
            if best is None or m > best[1]:
                from sklearn.metrics import average_precision_score
                pr = float(average_precision_score(yva, p))
                cm = ids.metrics(yva, p, w=a.w, th=th)
                best = (float(al), m, th, pr, cm["cost"])
        return best

    results = []

    def evaluate(tag, lkw, hkw):
        t = time.time()
        lgbm, hgb = ids.train_ensemble(Xtr, ytr, seed=42, lgb_kw=lkw, hgb_kw=hkw)
        p_lgb = lgbm.predict_proba(Xva)[:, 1]
        p_hgb = hgb.predict_proba(Xva)[:, 1]
        al, f1, th, pr, cost = score(p_lgb, p_hgb)
        results.append((f1, pr, cost, al, th, tag, time.time() - t))
        print(f"[tune ] {tag:<34} F1={f1:.4f} PR-AUC={pr:.4f} "
              f"cost={cost:,} (a={al:.1f} th={th:.3f}) "
              f"{results[-1][-1]:.0f}s")

    for name, kw in LGB_VARIANTS.items():
        evaluate(f"LGB {name} + HGB cur", kw, HGB_VARIANTS["cur(200,63,.08)"])
    for name, kw in HGB_VARIANTS.items():
        if name.startswith("cur"):
            continue
        evaluate(f"LGB cur + HGB {name}", LGB_VARIANTS["cur(400,64,.05,50)"], kw)

    results.sort(reverse=True, key=lambda r: r[0])
    print("\n[TUNE RANKING] by group-val F1-opt")
    print(f"  {'F1':>7} {'PR-AUC':>8} {'cost':>8} {'a':>4} {'th':>6}  config")
    for f1, pr, cost, al, th, tag, _ in results:
        print(f"  {f1:>7.4f} {pr:>8.4f} {cost:>8,} {al:>4.1f} {th:>6.3f}  {tag}")

    # combo: retrain top-2 LGB x top-2 HGB if they differ from what we ran
    print("\n[combo] combining best variants...")
    top_lgb = [r[5] for r in results]
    best_lgb_kw = None
    for name, kw in LGB_VARIANTS.items():
        if f"LGB {name} + HGB cur" == results[0][5]:
            best_lgb_kw = kw
    best_hgb_kw = None
    for name, kw in HGB_VARIANTS.items():
        if results[0][5].endswith(f"HGB {name}"):
            best_hgb_kw = kw
    if best_lgb_kw and best_hgb_kw and "cur" not in results[0][5]:
        evaluate("COMBO best-LGB + best-HGB", best_lgb_kw, best_hgb_kw)
        results.sort(reverse=True, key=lambda r: r[0])

    w = results[0]
    print(f"\n[WINNER] {w[5]}  F1={w[0]:.4f} a={w[3]:.1f} th={w[4]:.3f}")
    print("run fit --save with:")
    print(f"  .venv/bin/python run.py fit --save --alpha {w[3]:.1f} "
          f"--th {w[4]:.4f} <params from winner tag>")


# --------------------------------------------------------------- package
PKG_DIR = "final_submission"


def cmd_package(a):
    """(Re)build the organizer package from the saved artifact."""
    import platform
    from importlib.metadata import version

    art = joblib.load(ARTIFACT)
    os.makedirs(PKG_DIR, exist_ok=True)
    blob = {"lgbm": art.lgbm, "hgb": art.hgb, "cb": getattr(art, "cb", None),
            "maps": art.maps, "feature_cols": art.feature_cols,
            "threshold": art.threshold,
            "fe": art.meta.get("fe", True),
            "alpha": getattr(art, "alpha", 0.5), "meta": art.meta}
    joblib.dump(blob, f"{PKG_DIR}/model.joblib", compress=3)

    pkgs = ["scikit-learn", "lightgbm", "pandas", "numpy", "joblib", "scipy"]
    if blob["cb"] is not None:
        pkgs.append("catboost")
    with open(f"{PKG_DIR}/requirements.txt", "w") as f:
        f.write("\n".join(f"{p}=={version(p)}" for p in pkgs) + "\n")

    n_feat = len(art.feature_cols)
    has_cb = blob["cb"] is not None
    meta = {
        "team_name": "team_name",
        "architecture": (
            f"Ensemble: LightGBM ({art.meta.get('trees', '?')} trees, "
            f"leaves={art.meta.get('leaves', '?')}, lr={art.meta.get('lgb_lr', '?')}) "
            f"+ HistGradientBoosting ({art.meta.get('hgb_iter', '?')} iter, "
            f"leaves={art.meta.get('hgb_leaves', '?')}, lr={art.meta.get('hgb_lr', '?')})"
            + (f" + CatBoost (500 iter, depth=8, lr=0.05)"
               if has_cb else "")
            + (", blend weights 0.4/0.3/0.3 (lgb/hgb/cat)"
               if has_cb else f", blend alpha={art.alpha}")),
        "features": f"{n_feat} columns: 38 raw + engineered (log1p heavy-tail, "
                    "bidirectional ratios, rates, missingness flags) + 4 interaction "
                    "flags (fin_miss_svc, low_byte_udp, rst_no_svc, bytes_rate_low)",
        "decision_threshold": art.threshold,
        "threshold_rationale": art.meta.get(
            "threshold_rationale",
            "forced T=0.02 (cost-optimal under committee cost 40*FN+FP; "
            "kept per team decision)"),
        "cost_function": f"Cost = {art.meta.get('w', 40):.0f}*FN + FP "
                         "(committee update; retune: run.py fit --save --w <W>)",
        "training_data": "shared/data/train.csv (1,524,028 rows, group-hash split)",
        "validation": "group-pure 85/15 split + official validation.csv confirmation",
        "framework_versions": {p: version(p) for p in pkgs},
        "python": platform.python_version(),
        "model_params": {k: v for k, v in art.meta.items()
                         if k in ("alpha", "trees", "leaves", "lgb_lr", "mcs",
                                  "hgb_iter", "hgb_leaves", "hgb_lr",
                                  "cat", "fam_bal")},
    }
    with open(f"{PKG_DIR}/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[pkg  ] rebuilt {PKG_DIR}/ (threshold={art.threshold}, "
          f"feats={n_feat}, catboost={'on' if has_cb else 'off'})")


# --------------------------------------------------------------- lomo
def _quantile_transform(score, ref):
    """Map scores to their empirical quantile vs a reference set (monotonic)."""
    ref_sorted = np.sort(np.asarray(ref))
    return np.searchsorted(ref_sorted, np.asarray(score), side="left") / len(ref_sorted)


def cmd_lomo(a):
    from sklearn.metrics import average_precision_score
    tr = load_train(a.quick, dedupe=a.dedupe)
    families = sorted(c for c in tr["attack_cat"].unique() if c != "Normal")
    print(f"[lomo ] families: {families}")
    normals = tr[tr["Label"] == 0]
    print(f"[lomo ] normals available: {len(normals):,}")

    lkw, hkw = {}, {}
    if a.spw > 0:
        lkw["scale_pos_weight"] = a.spw
        hkw["class_weight"] = "balanced"

    rows = []
    for fam in families:
        held = tr[tr["attack_cat"] == fam]
        rest = tr[tr["attack_cat"] != fam]
        nsamp = normals.sample(n=min(400_000, len(normals)), random_state=0)
        eval_df = pd.concat([held, nsamp], ignore_index=True)

        Xtr, maps = ids.design_matrix(rest, fe=True, is_train=True)
        Xev, _ = ids.design_matrix(eval_df, maps=maps, fe=True)
        Xev = Xev[Xtr.columns]
        t = time.time()
        sw = ids.family_weights(rest) if a.fam_bal else None
        lgbm, hgb = ids.train_ensemble(Xtr, rest["Label"].values, seed=42,
                                       lgb_kw={"n_estimators": 250, **lkw},
                                       hgb_kw=hkw, sw=sw)
        p_sup = ids.ensemble_proba(lgbm, hgb, Xev)
        y = eval_df["Label"].values
        pr_sup = float(average_precision_score(y, p_sup))

        pr_anom = pr_b05 = pr_b025 = None
        if a.anom:
            from sklearn.ensemble import IsolationForest
            norm_tr = rest[rest["Label"] == 0].sample(
                n=min(300_000, int((rest["Label"] == 0).sum())), random_state=0)
            Xn, _ = ids.design_matrix(norm_tr, fe=True, is_train=False)
            Xn = Xn[Xtr.columns]
            iforest = IsolationForest(
                n_estimators=200, contamination="auto", max_samples=256,
                random_state=42, n_jobs=-1)
            iforest.fit(Xn)
            # anomaly: higher = more anomalous
            a_tr = -iforest.decision_function(Xn)
            a_ev = -iforest.decision_function(Xev[Xn.columns])
            q_anom = _quantile_transform(a_ev, a_tr)
            q_sup = _quantile_transform(p_sup, p_sup[y == 0])
            pr_anom = float(average_precision_score(y, q_anom))
            pr_b05 = float(average_precision_score(y, 0.5 * q_sup + 0.5 * q_anom))
            pr_b025 = float(average_precision_score(y, 0.75 * q_sup + 0.25 * q_anom))

        print(f"[lomo ] {fam:<12} n={len(held):>7,}  sup={pr_sup:.4f}  "
              + (f"anom={pr_anom:.4f}  blend50={pr_b05:.4f}  blend25={pr_b025:.4f}  "
                 if a.anom else "")
              + f"({time.time() - t:.0f}s)")
        rows.append((fam, len(held), pr_sup, pr_anom, pr_b05, pr_b025))

    print("\n[LOMO SUMMARY] unseen-family generalization (PR-AUC)")
    hdr = f"  {'family':<12} {'n':>7} {'sup':>8}"
    if a.anom:
        hdr += f" {'anom':>8} {'blend50':>8} {'blend25':>8}"
    print(hdr)
    for fam, n, s, an, b5, b25 in rows:
        line = f"  {fam:<12} {n:>7,} {s:>8.4f}"
        if a.anom:
            line += f" {an:>8.4f} {b5:>8.4f} {b25:>8.4f}"
        print(line)

    def agg(i):
        vals = [r[i] for r in rows if r[i] is not None]
        return (np.mean(vals), min(vals)) if vals else (None, None)
    for name, i in ([("sup", 2)] if not a.anom else
                    [("sup", 2), ("anom", 3), ("blend50", 4), ("blend25", 5)]):
        mean, worst = agg(i)
        if mean is not None:
            print(f"  {name:<8} mean={mean:.4f} worst={worst:.4f}")


# --------------------------------------------------------------- predict
def cmd_predict(a):
    art = joblib.load(a.model)
    df = ids.load(a.input, labelled=False)
    sub = art.submission(df)
    sub.to_csv(a.output, index=False)
    print(f"[pred ] {len(sub):,} rows -> {a.output} "
          f"(th={art.threshold:.4f}, attacks={int(sub['prediction'].sum()):,})")


# --------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--quick", action="store_true",
                       help="subsample for fast iteration")
        p.add_argument("--w", type=float, default=40.0,
                       help="false-negative cost weight (committee: 40*FN + FP)")
        p.add_argument("--dedupe", action="store_true",
                       help="drop exact duplicate feature rows from training")

    p = sub.add_parser("bar", help="reproduce baseline")
    common(p)

    p = sub.add_parser("fit", help="train ensemble + save artifact")
    common(p)
    p.add_argument("--trees", type=int, default=600)
    p.add_argument("--leaves", type=int, default=127)
    p.add_argument("--lgb-lr", type=float, default=0.03)
    p.add_argument("--mcs", type=int, default=50, help="min_child_samples")
    p.add_argument("--hgb-iter", type=int, default=200)
    p.add_argument("--hgb-leaves", type=int, default=63)
    p.add_argument("--hgb-lr", type=float, default=0.08)
    p.add_argument("--alpha", type=float, default=1.0,
                   help="LightGBM weight in ensemble blend")
    p.add_argument("--no-fe", action="store_true")
    p.add_argument("--cat", action=argparse.BooleanOptionalAction, default=True,
                   help="CatBoost 3rd blend model (default on)")
    p.add_argument("--fam-bal", action=argparse.BooleanOptionalAction, default=False,
                   help="attack-family reweighting (default off)")
    p.add_argument("--spw", type=float, default=0.0,
                   help="scale_pos_weight (0 = disabled)")
    p.add_argument("--th", type=float, default=None,
                   help="force artifact threshold (default: selected by --objective)")
    p.add_argument("--objective", choices=("f1", "cost"), default="f1",
                   help="automatic threshold objective (default: f1)")
    p.add_argument("--save", action="store_true")
    p.add_argument("--package", action="store_true",
                   help="rebuild final_submission/ after saving")

    p = sub.add_parser("tune", help="grid search hyperparameters + blend weight")
    common(p)
    p.add_argument("--no-fe", action="store_true")

    p = sub.add_parser("package", help="rebuild organizer package from artifact")
    common(p)

    p = sub.add_parser("lomo", help="leave-one-family-out CV")
    common(p)
    p.add_argument("--spw", type=float, default=0.0,
                   help="scale_pos_weight (0 = disabled)")
    p.add_argument("--fam-bal", action="store_true",
                   help="equalize mass across attack families")
    p.add_argument("--anom", action="store_true",
                   help="also evaluate IsolationForest anomaly blend")

    p = sub.add_parser("predict", help="generate submission")
    p.add_argument("--model", default=ARTIFACT)
    p.add_argument("--input", "-i", required=True)
    p.add_argument("--output", "-o", required=True)

    argv = sys.argv[1:] or ["fit", "--quick"]
    a = ap.parse_args(argv)
    {"bar": cmd_bar, "fit": cmd_fit, "lomo": cmd_lomo, "tune": cmd_tune,
     "package": cmd_package, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
