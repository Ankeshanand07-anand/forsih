#!/usr/bin/env python3
"""Run Person-A pipeline starting from an XGBoost-ready CSV file.

This is a reproducible, deterministic runner that implements downstream Person-A steps
from the preprocessed CSV onward.

It is intentionally conservative about assumptions: it will try to infer the timestamp
and PM2.5 target column from common names. If it cannot find them it will abort with
an informative error.
"""
import argparse
import os
import sys
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

COMMON_TIMESTAMP_COLS = ["datetime","timestamp","time","date","DateTime","DATE","Date"]
COMMON_TARGET_COLS = ["pm2.5","PM2.5","pm25","PM25","pm_2_5","target","y"]


def infer_columns(df: pd.DataFrame):
    ts_col = None
    for c in COMMON_TIMESTAMP_COLS:
        if c in df.columns:
            ts_col = c
            break
    # try columns that look like datetime (object dtype with parseable values)
    if ts_col is None:
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    pd.to_datetime(df[c].dropna().iloc[:20])
                    ts_col = c
                    break
                except Exception:
                    continue
    target_col = None
    for c in COMMON_TARGET_COLS:
        if c in df.columns:
            target_col = c
            break
    # fallback: numeric column with name containing 'pm' or 'pm2' or 'pm25'
    if target_col is None:
        for c in df.columns:
            lc = c.lower()
            if ("pm" in lc and any(ch.isdigit() for ch in lc)) or "pm25" in lc:
                if pd.api.types.is_numeric_dtype(df[c]):
                    target_col = c
                    break
    return ts_col, target_col


def chronological_split(df, ts_col, train_frac=0.70, val_frac=0.15):
    df_sorted = df.sort_values(ts_col).reset_index(drop=True)
    n = len(df_sorted)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    train = df_sorted.iloc[:i_train].copy()
    val = df_sorted.iloc[i_train:i_val].copy()
    test = df_sorted.iloc[i_val:].copy()
    return train, val, test


def make_lag_features(df, target_col, n_lags=24):
    X = pd.DataFrame(index=df.index)
    for lag in range(1, n_lags + 1):
        X[f"lag_{lag}"] = df[target_col].shift(lag)
    return X


def compute_persistence_errors(test_df, ts_col, target_col, horizons=[1,3,6,24,72]):
    # persistence: predict t+h = last observed value (t)
    res = []
    y = test_df[target_col].values
    n = len(y)
    for h in horizons:
        if h >= n:
            continue
        y_pred = y[:-h]
        y_true = y[h:]
        mae = mean_absolute_error(y_true, y_pred)
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        res.append({"horizon": h, "mae": float(mae), "rmse": float(rmse)})
    return pd.DataFrame(res)


def train_ridge_baseline(train, val, test, ts_col, target_col, outdir, lags=24):
    # Create lag features on the concatenated set to ensure consistent columns
    concat = pd.concat([train, val, test], axis=0).reset_index(drop=True)
    X = make_lag_features(concat, target_col, n_lags=lags)
    y = concat[target_col]
    # drop rows with NaNs introduced by lagging
    valid = X.dropna().index
    X_valid = X.loc[valid]
    y_valid = y.loc[valid]
    # determine split indices
    n_train = len(train)
    n_val = len(val)
    # Because we shifted, valid indices start at "lags"
    # Map original positions
    train_idx = valid[valid < n_train].tolist()
    val_idx = valid[(valid >= n_train) & (valid < n_train + n_val)].tolist()
    test_idx = valid[valid >= n_train + n_val].tolist()
    if len(train_idx) == 0 or len(test_idx) == 0:
        warnings.warn("Not enough data after lagging to fit Ridge baseline. Skipping.")
        return None
    X_train = X_valid.loc[train_idx]
    y_train = y_valid.loc[train_idx]
    X_test = X_valid.loc[test_idx]
    y_test = y_valid.loc[test_idx]
    model = Ridge(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    # Save model
    joblib.dump(model, os.path.join(outdir, "models", "ridge_baseline.joblib"))
    return {"mae": float(mae), "rmse": float(rmse)}


def data_quality_report(df, ts_col, outpath):
    summary = []
    for c in df.columns:
        summary.append({
            "column": c,
            "dtype": str(df[c].dtype),
            "n_missing": int(df[c].isna().sum()),
            "n_unique": int(df[c].nunique(dropna=True)) if df[c].nunique(dropna=True) < 1e6 else -1
        })
    # time range
    try:
        tmin = df[ts_col].min()
        tmax = df[ts_col].max()
    except Exception:
        tmin = None
        tmax = None
    meta = {"time_min": str(tmin), "time_max": str(tmax), "n_rows": int(len(df))}
    rep = pd.DataFrame(summary)
    rep.to_csv(outpath, index=False)
    return meta


def ensure_dirs(outdir):
    Path(outdir).mkdir(parents=True, exist_ok=True)
    Path(outdir, "models").mkdir(parents=True, exist_ok=True)
    Path(outdir, "final").mkdir(parents=True, exist_ok=True)
    Path(outdir, "reports").mkdir(parents=True, exist_ok=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to xgboost_ready_for_personA.csv or .csv.gz")
    p.add_argument("--outdir", required=True, help="Output directory for PersonA_COMPLETE")
    p.add_argument("--lags", type=int, default=24, help="Number of lag features to create for baseline")
    args = p.parse_args()

    inp = Path(args.input)
    outdir = Path(args.outdir)
    ensure_dirs(outdir)

    if not inp.exists():
        print(f"Input file not found: {inp}")
        sys.exit(1)

    print("Loading input CSV (may be gzipped)...")
    df = pd.read_csv(inp, low_memory=False)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")

    ts_col, target_col = infer_columns(df)
    if ts_col is None:
        print("Could not infer a timestamp column. Please provide a CSV with a timestamp column named one of:", COMMON_TIMESTAMP_COLS)
        sys.exit(1)
    if target_col is None:
        print("Could not infer a PM2.5 target column. Please ensure the CSV contains PM2.5 column (pm25, PM2.5, etc.)")
        sys.exit(1)
    print(f"Inferred timestamp column: {ts_col}")
    print(f"Inferred target column: {target_col}")

    # parse timestamp
    df[ts_col] = pd.to_datetime(df[ts_col], errors='coerce')
    # drop rows without timestamp
    df = df.dropna(subset=[ts_col]).reset_index(drop=True)

    # Save an uncompressed copy of xgboost_ready for downstream reproducibility
    df.to_csv(outdir / "final" / "xgboost_ready_for_personA.csv", index=False)

    # Data quality report
    meta = data_quality_report(df, ts_col, outdir / "reports" / "data_quality_report.csv")
    print("Wrote data quality report; dataset time range:", meta.get("time_min"), "->", meta.get("time_max"))

    # Chronological splits
    train, val, test = chronological_split(df, ts_col)
    train.to_csv(outdir / "final" / "train.csv", index=False)
    val.to_csv(outdir / "final" / "validation.csv", index=False)
    test.to_csv(outdir / "final" / "test.csv", index=False)
    print(f"Train/Val/Test sizes: {len(train)}/{len(val)}/{len(test)}")

    # Baselines: persistence
    persistence_results = compute_persistence_errors(test, ts_col, target_col)
    persistence_results.to_csv(outdir / "reports" / "persistence_baseline_results.csv", index=False)
    print("Wrote persistence baseline results")

    # Ridge baseline using lag features
    ridge_res = train_ridge_baseline(train, val, test, ts_col, target_col, str(outdir), lags=args.lags)
    if ridge_res is not None:
        pd.DataFrame([ridge_res]).to_csv(outdir / "reports" / "ridge_baseline_results.csv", index=False)
        print("Trained and saved Ridge baseline")
    else:
        print("Ridge baseline skipped due to insufficient data after lagging")

    # Sequence file for Person B (past 72h -> next 72h)
    from personA.processing.sequence_generator import create_sequence_base
    seq_path = outdir / "final" / "sequence_base_for_personB.csv"
    create_sequence_base(df, ts_col, target_col, past_hours=72, future_hours=72, outpath=seq_path)
    print("Wrote sequence_base_for_personB.csv")

    # Save a small manifest
    manifest = {
        "input_used": str(inp),
        "n_rows": int(len(df)),
        "time_min": meta.get("time_min"),
        "time_max": meta.get("time_max"),
        "timestamp_column": ts_col,
        "target_column": target_col
    }
    pd.Series(manifest).to_csv(outdir / "final" / "personA_manifest.csv")
    print("Person-A pipeline complete. Outputs in", outdir)


if __name__ == "__main__":
    main()
