#!/usr/bin/env python3
"""
final_submission/predict.py
===========================
Standard inference script for the ML Hackathon intrusion detection submission.

Run:
    python predict.py --input path/to/test.csv --output predictions.csv

Input : headerless CSV with the 38 competition feature columns (in order).
Output: CSV with row_id,prediction,probability (row_id = 1..N, prediction in {0,1}).

The artifact (model.joblib) contains: LightGBM + HistGradientBoosting +
CatBoost models, categorical encoders, engineered feature spec (including
interaction flags), and the decision threshold.
Pipeline is fully deterministic — repeated runs are bit-identical.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]
MISSING_FLAG_SRC = ["is_ftp_login", "ct_ftp_cmd", "ct_flw_http_mthd"]
LOG1P = [
    "dur", "Sload", "Dload", "sbytes", "dbytes", "Sjit", "Djit", "Sintpkt",
    "Dintpkt", "stcpb", "dtcpb", "res_bdy_len", "tcprtt", "synack", "ackdat",
    "Spkts", "Dpkts", "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
]


def add_interactions(X, df, N):
    # Interaction features for stealth/FIN-service gap
    X["fin_miss_svc"] = ((df["state"].astype(str).str.strip() == "FIN") &
                         df["service"].astype(str).str.strip().isin(["", " ", "-"])).astype(np.float32)
    X["low_byte_udp"] = ((N["dur"] < 1.0) & (N["Spkts"] < 3) &
                         (df["proto"].astype(str).str.strip() == "udp")).astype(np.float32)
    X["rst_no_svc"] = ((df["state"].astype(str).str.strip() == "RST") &
                       df["service"].astype(str).str.strip().isin(["", " ", "-"])).astype(np.float32)
    X["bytes_rate_low"] = (N["sbytes"] / (N["dur"].clip(lower=1.0) + 1e-3) < 100).astype(np.float32)
    return X


def design_matrix(df, maps, fe=True):
    """Rebuild the exact training design matrix from raw string columns."""
    X = pd.DataFrame(index=df.index)

    if fe:
        for c in MISSING_FLAG_SRC:
            X[f"m_{c}"] = df[c].isin(["", " "]).astype(np.float32)

    for c in CATEGORICAL_COLS:
        X[c] = df[c].astype(str).map(maps[c]).fillna(len(maps[c])).astype(np.int32)

    N = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in NUMERIC_COLS},
                     index=df.index).astype(np.float32)
    for c in NUMERIC_COLS:
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
        add_interactions(X, df, N)

    return X


def main():
    parser = argparse.ArgumentParser(description="Intrusion detection model inference")
    parser.add_argument("--input", "-i", type=str, required=True,
                        help="Path to input unlabelled CSV (38 feature columns)")
    parser.add_argument("--output", "-o", type=str, required=True,
                        help="Path to save output prediction CSV")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Path to model.joblib (default: next to this script)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] Error: input file not found: {args.input}")
        sys.exit(1)

    model_path = args.model or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "model.joblib")
    if not os.path.exists(model_path):
        print(f"[!] Error: model artifact not found: {model_path}")
        sys.exit(1)

    print(f"[*] Loading model artifact: {model_path} ...")
    blob = joblib.load(model_path)
    lgbm, hgb = blob["lgbm"], blob["hgb"]
    cb = blob.get("cb", None)
    maps, feature_cols = blob["maps"], blob["feature_cols"]
    threshold = float(blob.get("threshold", 0.5))
    fe = bool(blob.get("fe", True))
    alpha = float(blob.get("alpha", 0.5))

    print(f"[*] Reading test features: {args.input} ...")
    # Positional read: takes the first 38 columns. Works for the 38-column
    # challenge/final files AND for 40-column labelled files (validation),
    # without pandas promoting leading columns to the index.
    raw = pd.read_csv(args.input, header=None, dtype=str, keep_default_na=False)
    if len(raw) and str(raw.iloc[0, 0]).strip().lower() in ("proto", '"proto"'):
        print("    Note: header row detected and skipped.")
        raw = raw.iloc[1:].reset_index(drop=True)
    if raw.shape[1] < len(FEATURE_NAMES):
        print(f"[!] Error: expected {len(FEATURE_NAMES)} feature columns, "
              f"got {raw.shape[1]}")
        sys.exit(1)
    if raw.shape[1] > len(FEATURE_NAMES):
        print(f"    Note: file has {raw.shape[1]} columns; using the first "
              f"{len(FEATURE_NAMES)} as features.")
    df = raw.iloc[:, :len(FEATURE_NAMES)].copy()
    df.columns = FEATURE_NAMES
    print(f"    Loaded {len(df):,} records.")

    print("[*] Building engineered feature matrix ...")
    X = design_matrix(df, maps, fe=fe)[feature_cols]

    print(f"[*] Predicting (threshold = {threshold:.4f}) ...")
    if cb is not None:
        probs = (0.4 * lgbm.predict_proba(X)[:, 1] +
                 0.3 * hgb.predict_proba(X)[:, 1] +
                 0.3 * cb.predict_proba(X)[:, 1])
    else:
        probs = alpha * lgbm.predict_proba(X)[:, 1] + (1.0 - alpha) * hgb.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)

    out = pd.DataFrame({
        "row_id": np.arange(1, len(preds) + 1),
        "prediction": preds,
        "probability": np.round(probs, 6),
    })
    out.to_csv(args.output, index=False)
    print(f"[+] Saved {len(out):,} predictions to: {args.output}")
    print(f"    attacks flagged: {int(preds.sum()):,} ({preds.mean():.2%})")


if __name__ == "__main__":
    main()
