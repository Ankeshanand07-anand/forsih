"""Small utility to generate past->future sequences for Person B.

Generates rows with columns:
- start_timestamp
- end_timestamp
- past_0 ... past_{past_hours-1}: values of target for past hours (most recent last)
- fut_0 ... fut_{future_hours-1}: target values for future (if present) or NaN

Output is saved as a CSV and can be used by Person B to build sequence models.
"""
from pathlib import Path
import pandas as pd
import numpy as np


def create_sequence_base(df, ts_col, target_col, past_hours=72, future_hours=72, outpath=None):
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df[[ts_col, target_col]].copy()
    df = df.dropna(subset=[ts_col])
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.set_index(ts_col).asfreq("H")
    # Ensure numeric
    series = df[target_col].astype(float)
    rows = []
    idx = series.index
    for i in range(past_hours, len(series) - future_hours + 1):
        past_vals = series.iloc[i - past_hours:i].values
        future_vals = series.iloc[i:i + future_hours].values
        row = {}
        row["start_timestamp"] = idx[i - past_hours]
        row["reference_timestamp"] = idx[i]
        # past_0 oldest ... past_{past_hours-1} most recent
        for p in range(past_hours):
            row[f"past_{p}"] = past_vals[p]
        for f in range(future_hours):
            row[f"fut_{f}"] = future_vals[f]
        rows.append(row)
    out = pd.DataFrame(rows)
    if outpath is not None:
        Path(outpath).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(outpath, index=False)
    return out


if __name__ == "__main__":
    # quick smoke test
    dates = pd.date_range("2020-01-01", periods=200, freq="H")
    s = pd.Series(np.random.randn(len(dates)), index=dates, name="pm25")
    df = s.reset_index()
    df.columns = ["datetime", "pm25"]
    create_sequence_base(df, "datetime", "pm25", past_hours=24, future_hours=24, outpath="sequence_test.csv")
