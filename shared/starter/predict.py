#!/usr/bin/env python3
"""
starter/predict.py
==================
ML Hackathon  -  Network Intrusion Detection
Standard Model Inference Script Template

Participants: You can use or adapt this script for your final model submission.
Organizers will test your model by running:
    python predict.py --input path/to/test_features.csv --output predictions.csv

This script:
1. Loads your saved model artifact (e.g. baseline_model.joblib).
2. Reads the unlabelled input CSV containing the 38 competition features.
3. Applies your feature preprocessing.
4. Generates predictions using your chosen decision threshold.
5. Saves the output in the required CSV format (row_id,prediction,probability).
"""

import os
import sys
import argparse
import joblib
import numpy as np
import pandas as pd

FEATURE_NAMES = [
    "proto", "state", "dur", "sbytes", "dbytes", "sloss", "dloss", "service",
    "Sload", "Dload", "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb",
    "smeansz", "dmeansz", "trans_depth", "res_bdy_len", "Sjit", "Djit",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd", "ct_srv_src", "ct_srv_dst",
    "ct_dst_ltm", "ct_src_ltm", "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm"
]
CATEGORICAL_COLS = ["proto", "state", "service"]
NUMERIC_COLS = [c for c in FEATURE_NAMES if c not in CATEGORICAL_COLS]


def preprocess(df, cat_levels):
    X = pd.DataFrame(index=df.index)
    for c in CATEGORICAL_COLS:
        mapping = cat_levels.get(c, {})
        oov_idx = len(mapping)
        X[c] = df[c].astype(str).map(mapping).fillna(oov_idx).astype(int)
    for c in NUMERIC_COLS:
        X[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(np.float32)
    return X


def main():
    parser = argparse.ArgumentParser(description="Model Inference Script")
    parser.add_argument("--input", "-i", type=str, required=True, help="Path to input unlabelled CSV (38 feature columns)")
    parser.add_argument("--output", "-o", type=str, required=True, help="Path to save output prediction CSV")
    parser.add_argument("--model", "-m", type=str, default="baseline_model.joblib", help="Path to saved model artifact")
    args = parser.parse_args()

    model_path = args.model
    if not os.path.exists(model_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        for cand in [os.path.join(script_dir, model_path), os.path.join(script_dir, "..", model_path)]:
            if os.path.exists(cand):
                model_path = cand
                break

    if not os.path.exists(model_path):
        print(f"[!] Error: Model artifact not found: {args.model}")
        sys.exit(1)

    print(f"[*] Loading model artifact: {model_path} ...")
    package = joblib.load(model_path)
    model = package["model"]
    cat_levels = package.get("cat_levels", {})
    threshold = package.get("best_threshold", 0.5)

    print(f"[*] Reading test features: {args.input} ...")
    test_df = pd.read_csv(
        args.input,
        header=None,
        names=FEATURE_NAMES,
        dtype=str,
        keep_default_na=False
    )
    print(f"    Loaded {len(test_df):,} records.")

    print("[*] Running feature preprocessing ...")
    X_test = preprocess(test_df, cat_levels)

    print("[*] Generating predictions (Threshold = {:.4f}) ...".format(threshold))
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    sub_df = pd.DataFrame({
        "row_id": np.arange(1, len(preds) + 1),
        "prediction": preds,
        "probability": np.round(probs, 6)
    })
    sub_df.to_csv(args.output, index=False)
    print(f"[+] Predictions saved successfully to: {args.output}")
    print(f"    Total predictions: {len(sub_df):,}")


if __name__ == "__main__":
    main()
