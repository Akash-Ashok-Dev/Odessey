#!/usr/bin/env python3
"""
ids.py — core library for Network Intrusion Detection hackathon.
Feature design, models, metrics, threshold optimization, artifact class.
"""
import numpy as np
import pandas as pd

FEATS = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]
CATS = ["proto", "state", "service"]
NUM = [c for c in FEATS if c not in CATS]
COLS = FEATS + ["attack_cat", "Label"]

MISSING_FLAG_SRC = ["is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd"]

LOG1P = [
    "dur", "Sload", "Dload", "sbytes", "dbytes", "Sjit", "Djit", "Sintpkt",
    "Dintpkt", "stcpb", "dtcpb", "res_bdy_len", "tcprtt", "synack", "ackdat",
    "Spkts", "Dpkts", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]


# ---------------------------------------------------------------- loading
def load(path, labelled=True, nrows=None):
    """
    Load headerless CSV. Uses positional column selection so a 40-column
    labelled file (38 features + attack_cat + Label) and a 38-column test
    file both read correctly (never lets pandas promote columns to index).
    """
    df = pd.read_csv(path, header=None, dtype=str, keep_default_na=False, nrows=nrows)
    if len(df) and str(df.iloc[0, 0]).strip().lower() in ("proto", '"proto"'):
        df = df.iloc[1:].reset_index(drop=True)
    if df.shape[1] < len(FEATS):
        raise ValueError(
            f"{path}: expected >= {len(FEATS)} columns, got {df.shape[1]}")
    if labelled:
        if df.shape[1] < len(COLS):
            raise ValueError(
                f"{path}: labelled file needs {len(COLS)} columns, got {df.shape[1]}")
        df = df.iloc[:, :len(COLS)].copy()
        df.columns = COLS
        df["attack_cat"] = df["attack_cat"].str.strip().replace({"": "Normal"})
        df["Label"] = pd.to_numeric(df["Label"], errors="coerce").fillna(0).astype(int)
    else:
        df = df.iloc[:, :len(FEATS)].copy()
        df.columns = FEATS
    return df


def load_labelled_with_dupkey(path, nrows=None):
    """Load labelled data plus a dedup/group hash key (fast, C-level hashing)."""
    df = load(path, labelled=True, nrows=nrows)
    df["_g"] = pd.util.hash_pandas_object(df[FEATS], index=False).astype("uint64")
    return df


def dedupe(df):
    """Drop exact duplicate feature rows (keep first)."""
    before = len(df)
    df = df.drop_duplicates(subset=FEATS, keep="first").copy()
    return df, before - len(df)


def label_conflicts(df):
    """Rows with identical features but opposing labels (label noise)."""
    if "_g" not in df.columns:
        return 0
    g = df.groupby("_g", sort=False)["Label"]
    bad = (g.nunique() > 1)
    return int(df["_g"].isin(bad[bad].index).sum())


def group_split(df, val_pct=15, seed=0):
    """
    Deterministic group split by feature-hash: identical flows never straddle.
    Returns (train_df, val_df).
    """
    if "_g" not in df.columns:
        df = df.copy()
        df["_g"] = pd.util.hash_pandas_object(df[FEATS], index=False).astype("uint64")
    bucket = (df["_g"] + np.uint64(seed)) % np.uint64(1000)
    val_mask = bucket < np.uint64(val_pct * 10)
    return df[~val_mask].copy(), df[val_mask].copy()


# ---------------------------------------------------------------- features
def build_maps(df):
    return {c: {v: i for i, v in enumerate(sorted(df[c].astype(str).unique()))}
            for c in CATS}


def design_matrix(df, maps=None, fe=True, is_train=False):
    """
    Build model matrix: integer-coded categoricals (OOV = len(map)) and
    float32 numerics with NaN preserved (both models handle NaN natively).
    With fe=True adds engineered features (logs, ratios, rates, missingness).
    """
    if maps is None or is_train:
        maps = build_maps(df)
    X = pd.DataFrame(index=df.index)

    if fe:
        for c in MISSING_FLAG_SRC:
            X[f"m_{c}"] = df[c].isin(["", " "]).astype(np.float32)

    for c in CATS:
        X[c] = df[c].astype(str).map(maps[c]).fillna(len(maps[c])).astype(np.int32)

    N = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in NUM},
                     index=df.index).astype(np.float32)
    for c in NUM:
        X[c] = N[c]

    if fe:
        s, d, dur = N["sbytes"], N["dbytes"], N["dur"]
        sp, dp = N["Spkts"], N["Dpkts"]
        X["s2d_bytes"] = s / (d + 1.0)
        X["byte_asym"] = (s - d) / (s + d + 1.0)
        X["pkt_ratio"] = sp / (dp + 1.0)
        X["pkt_asym"] = (sp - dp) / (sp + dp + 1.0)
        X["total_bytes"] = s + d
        X["total_pkts"] = sp + dp
        X["loss_ratio"] = (N["sloss"] + N["dloss"]) / (sp + dp + 1.0)
        X["sloss_rate"] = N["sloss"] / (sp + 1.0)
        X["dloss_rate"] = N["dloss"] / (dp + 1.0)
        X["bytes_rate"] = (s + d) / (dur + 1e-3)
        X["pkts_rate"] = (sp + dp) / (dur + 1e-3)
        X["avg_pkt"] = (s + d) / (sp + dp + 1.0)
        X["jit_ratio"] = N["Sjit"] / (N["Djit"] + 1.0)
        X["intpkt_ratio"] = N["Sintpkt"] / (N["Dintpkt"] + 1.0)
        X["ct_ratio"] = N["ct_src_ltm"] / (N["ct_dst_ltm"] + 1.0)
        X["srv_ratio"] = N["ct_srv_src"] / (N["ct_srv_dst"] + 1.0)
        X["win_ratio"] = N["swin"] / (N["dwin"] + 1.0)
        X["durb"] = (dur > 0).astype(np.float32)
        X["http_txn"] = (N["trans_depth"] > 0).astype(np.float32)
        X["has_body"] = (N["res_bdy_len"] > 0).astype(np.float32)
        for c in LOG1P:
            X[f"lg_{c}"] = np.log1p(N[c].clip(lower=0.0))
        X["lg_total_bytes"] = np.log1p((s + d).clip(lower=0.0))

    return X, maps


# ---------------------------------------------------------------- models
def cat_idx(X, cols=CATS):
    return [list(X.columns).index(c) for c in cols]


def family_weights(df):
    """
    Equalize mass across attack families so Generic's 87% share doesn't
    drown out minority families. Negatives keep weight 1. Mean weight = 1.
    """
    lab = df["Label"].to_numpy()
    cats = df["attack_cat"].to_numpy()
    w = np.ones(len(df), dtype=np.float64)
    pos = lab == 1
    fams, counts = np.unique(cats[pos], return_counts=True)
    k = len(fams)
    total = int(counts.sum())
    for f, c in zip(fams, counts):
        m = pos & (cats == f)
        w[m] = total / (k * c)
    return w * (len(w) / w.sum())


def train_ensemble(X, y, seed=42, lgb_kw=None, hgb_kw=None, sw=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import lightgbm as lgb

    ci = cat_idx(X)
    lk = dict(n_estimators=400, num_leaves=64, learning_rate=0.05,
              min_child_samples=50, subsample=0.9, subsample_freq=1,
              colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, verbose=-1)
    hk = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=63,
              min_samples_leaf=40, l2_regularization=1.0, random_state=seed)
    if lgb_kw:
        lk.update(lgb_kw)
    if hgb_kw:
        hk.update(hgb_kw)

    lgbm = lgb.LGBMClassifier(**lk)
    lgbm.fit(X, y, categorical_feature=ci, sample_weight=sw)
    hgb = HistGradientBoostingClassifier(categorical_features=ci, **hk)
    hgb.fit(X, y, sample_weight=sw)
    return lgbm, hgb


def ensemble_proba(lgbm, hgb, X):
    return 0.5 * lgbm.predict_proba(X)[:, 1] + 0.5 * hgb.predict_proba(X)[:, 1]


# ---------------------------------------------------------------- metrics
def metrics(y, p, w=20.0, th=0.5):
    from sklearn.metrics import (f1_score, precision_score, recall_score,
                                 average_precision_score, confusion_matrix)
    yh = (p >= th).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yh, labels=[0, 1]).ravel()
    return {
        "th": float(th),
        "f1": float(f1_score(y, yh, zero_division=0)),
        "pr_auc": float(average_precision_score(y, p)),
        "prec": float(precision_score(y, yh, zero_division=0)),
        "rec": float(recall_score(y, yh, zero_division=0)),
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "cost": int(w * fn + fp),
    }


def best_threshold(y, p, w=20.0, n_cand=1001):
    """Exact cost-optimal threshold via sorted cumulative counts."""
    ps = np.asarray(p, dtype=np.float64)
    ys = np.asarray(y, dtype=np.int64)
    order = np.argsort(ps, kind="mergesort")
    ps_s = ps[order]
    cum = np.concatenate([[0], np.cumsum(ys[order])])
    n = len(ps)
    P = int(cum[-1])
    cand = np.unique(np.concatenate([
        np.quantile(ps, np.linspace(0, 1, n_cand)),
        np.linspace(0.01, 0.99, 99),
    ]))
    idx = np.searchsorted(ps_s, cand, side="left")
    tp = P - cum[idx]
    fn = P - tp
    fp = (n - idx) - tp
    cost = w * fn + fp
    i = int(np.argmin(cost))
    tp_i, fp_i, fn_i = int(tp[i]), int(fp[i]), int(fn[i])
    return float(cand[i]), {
        "th": float(cand[i]), "f1": float(_f1(tp_i, fp_i, fn_i)),
        "pr_auc": float(average_precision_safe(ys, ps)),
        "prec": float(tp_i / max(tp_i + fp_i, 1)),
        "rec": float(tp_i / max(tp_i + fn_i, 1)),
        "tp": tp_i, "fp": fp_i, "fn": fn_i,
        "tn": int(n - tp_i - fp_i - fn_i), "cost": int(cost[i]),
    }


def _f1(tp, fp, fn):
    return 2 * tp / max(2 * tp + fp + fn, 1)


def average_precision_safe(y, p):
    from sklearn.metrics import average_precision_score
    try:
        return float(average_precision_score(y, p))
    except Exception:
        return 0.0


def report(y, p, cats=None, w=20.0, th=0.5, title=""):
    m = metrics(y, p, w=w, th=th)
    print(f"\n--- {title} (th={th:.4f}, w={w:.0f}) ---")
    print(f"  F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}  "
          f"P={m['prec']:.4f}  R={m['rec']:.4f}")
    print(f"  TP={m['tp']:,} FP={m['fp']:,} TN={m['tn']:,} FN={m['fn']:,}  "
          f"COST={m['cost']:,}")
    if cats is not None:
        cats = np.asarray(cats)
        yh = (p >= th).astype(int)
        parts = []
        for c in sorted(np.unique(cats)):
            mask = cats == c
            if c == "Normal":
                parts.append(f"Normal:{np.mean(yh[mask] == 0):.3f}")
            else:
                parts.append(f"{c}:{np.mean(yh[mask] == 1):.3f} (n={mask.sum()})")
        print("  per-cat: " + "  ".join(parts))
    return m


# ---------------------------------------------------------------- artifact
class Artifact:
    """Everything predict.py needs: maps, column order, models, threshold."""

    def __init__(self, maps, models, feature_cols, threshold, meta=None):
        self.maps = maps
        self.lgbm, self.hgb = models
        self.feature_cols = list(feature_cols)
        self.threshold = float(threshold)
        self.meta = meta or {}

    def transform(self, df, fe=True):
        X, _ = design_matrix(df, maps=self.maps, fe=fe)
        return X[self.feature_cols]

    def predict_proba(self, df):
        X = self.transform(df)
        return ensemble_proba(self.lgbm, self.hgb, X)

    def predict(self, df):
        return (self.predict_proba(df) >= self.threshold).astype(int)

    def submission(self, df):
        p = self.predict_proba(df)
        pred = (p >= self.threshold).astype(int)
        return pd.DataFrame({
            "row_id": np.arange(1, len(pred) + 1),
            "prediction": pred,
            "probability": np.round(p, 6),
        })
