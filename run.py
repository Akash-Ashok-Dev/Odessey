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
    lkw = {"n_estimators": a.trees}
    hkw = {}
    if a.spw > 0:
        lkw["scale_pos_weight"] = a.spw
        hkw["class_weight"] = "balanced"
        print(f"[train] positive class weighting: scale_pos_weight={a.spw}")
    t = time.time()
    lgbm, hgb = ids.train_ensemble(Xtr, ytr, seed=42, lgb_kw=lkw, hgb_kw=hkw)
    print(f"[train] ensemble: {time.time() - t:.1f}s")
    imp = sorted(zip(Xtr.columns, lgbm.feature_importances_), key=lambda z: -z[1])
    print("[lgb  ] top features: " + ", ".join(f"{c}:{v}" for c, v in imp[:12]))

    probs = ids.ensemble_proba(lgbm, hgb, Xva)
    ids.report(yva, probs, cats=va_g["attack_cat"].values, w=a.w, th=0.5,
               title="ENSEMBLE | group-val @0.5")
    if a.th is None:
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
    po = ids.ensemble_proba(lgbm, hgb, Xo)
    ids.report(vo["Label"].values, po, cats=vo["attack_cat"].values, w=a.w,
               th=0.5, title="ENSEMBLE | OFFICIAL val @0.5")
    ids.report(vo["Label"].values, po, cats=vo["attack_cat"].values, w=a.w,
               th=th, title="ENSEMBLE | OFFICIAL val @chosen threshold")
    op_table(vo["Label"].values, po, w=a.w, title="OFFICIAL val")
    op_table(yva, probs, w=a.w, title="group-val")

    if a.save:
        # final artifact trained on ALL deduped train (threshold from group-val)
        Xf, mapsf = ids.design_matrix(tr, fe=not a.no_fe, is_train=True)
        t = time.time()
        flgb, fhgb = ids.train_ensemble(Xf, tr["Label"].values, seed=42,
                                        lgb_kw=lkw, hgb_kw=hkw)
        print(f"[final] retrain on full train: {time.time() - t:.1f}s")
        art = ids.Artifact(mapsf, (flgb, fhgb), Xf.columns, th,
                           meta={"w": a.w, "fe": not a.no_fe, "trees": a.trees,
                                 "train_rows": len(tr)})
        os.makedirs(os.path.dirname(ARTIFACT), exist_ok=True)
        joblib.dump(art, ARTIFACT)
        print(f"[save ] artifact -> {ARTIFACT} (threshold={th:.4f})")


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
        p.add_argument("--w", type=float, default=20.0,
                       help="false-negative cost weight")
        p.add_argument("--dedupe", action="store_true",
                       help="drop exact duplicate feature rows from training")

    p = sub.add_parser("bar", help="reproduce baseline")
    common(p)

    p = sub.add_parser("fit", help="train ensemble + save artifact")
    common(p)
    p.add_argument("--trees", type=int, default=400)
    p.add_argument("--no-fe", action="store_true")
    p.add_argument("--spw", type=float, default=0.0,
                   help="scale_pos_weight (0 = disabled)")
    p.add_argument("--th", type=float, default=None,
                   help="force artifact threshold (default: group-val cost-opt)")
    p.add_argument("--save", action="store_true")

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

    a = ap.parse_args()
    {"bar": cmd_bar, "fit": cmd_fit, "lomo": cmd_lomo,
     "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
