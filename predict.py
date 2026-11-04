"""
predict.py
----------
Classify heartbeats with the trained model.

Usage:
    python predict.py --make_sample                # creates beats.csv from record 100
    python predict.py --csv beats.csv              # classify beats from a CSV file
    python predict.py --record 100                 # classify a dataset record
    python predict.py --csv beats.csv --out result.csv

CSV format (one row = one beat):
    s0, s1, ... s199   -> 200 filtered ECG samples (R peak at s90)   [required]
    pre_rr, post_rr, avg_rr -> RR intervals in seconds               [optional]
    (any other column, e.g. true_label, is ignored)
If the RR columns are missing, average RR values from training are used.
"""
import argparse

import numpy as np
import pandas as pd

from src.data_loader import CLASS_DESCRIPTIONS, DATA_DIR, find_records
from src.model import load_model, predict_beats
from src.preprocessing import BEAT_LEN, RR_COLUMNS, build_record_dataset

SIGNAL_COLUMNS = [f"s{i}" for i in range(BEAT_LEN)]


def read_beats_csv(path):
    df = pd.read_csv(path)
    if all(c in df.columns for c in SIGNAL_COLUMNS):
        beats = df[SIGNAL_COLUMNS].to_numpy(dtype=float)
    else:
        numeric = df.select_dtypes("number").drop(columns=RR_COLUMNS, errors="ignore")
        if numeric.shape[1] != BEAT_LEN:
            raise ValueError(
                f"Expected {BEAT_LEN} signal columns (s0..s{BEAT_LEN - 1}), found {numeric.shape[1]}."
            )
        beats = numeric.to_numpy(dtype=float)
    rr = df[RR_COLUMNS].to_numpy(dtype=float) if all(c in df.columns for c in RR_COLUMNS) else None
    return df, beats, rr


def make_sample(record="100", n_beats=20, out="beats.csv"):
    records = find_records()
    if record not in records:
        raise SystemExit(f"Record {record} not found in {DATA_DIR}")
    beats, rr, labels = build_record_dataset(records[record])
    n = min(n_beats, len(labels))
    df = pd.DataFrame(beats[:n], columns=SIGNAL_COLUMNS)
    for i, col in enumerate(RR_COLUMNS):
        df[col] = rr[:n, i]
    df["true_label"] = labels[:n]
    df.to_csv(out, index=False)
    print(f"Sample file created: {out} ({n} beats from record {record})")


def main():
    parser = argparse.ArgumentParser(description="ECG beat prediction")
    parser.add_argument("--csv", help="CSV file containing beats")
    parser.add_argument("--record", help="classify beats of a dataset record, e.g. 100")
    parser.add_argument("--n", type=int, default=None, help="limit number of beats (record mode)")
    parser.add_argument("--out", default="predictions.csv", help="output CSV file")
    parser.add_argument("--make_sample", action="store_true", help="create beats.csv sample file")
    args = parser.parse_args()

    if args.make_sample:
        make_sample(args.record or "100")
        return
    if not args.csv and not args.record:
        parser.print_help()
        return

    bundle = load_model()

    if args.csv:
        df, beats, rr = read_beats_csv(args.csv)
        pred, proba = predict_beats(bundle, beats, rr)
        out = df.copy()
    else:
        records = find_records(exclude_paced=False)
        if args.record not in records:
            raise SystemExit(f"Record {args.record} not found in {DATA_DIR}")
        beats, rr, labels = build_record_dataset(records[args.record])
        if args.n:
            beats, rr, labels = beats[: args.n], rr[: args.n], labels[: args.n]
        pred, proba = predict_beats(bundle, beats, rr)
        out = pd.DataFrame({"true_label": labels})

    out["predicted_class"] = pred
    out["description"] = [CLASS_DESCRIPTIONS[p] for p in pred]
    out["confidence"] = proba.max(axis=1).round(3)
    out = out.drop(columns=[c for c in SIGNAL_COLUMNS if c in out.columns])
    out.to_csv(args.out, index=False)

    print(out.head(20).to_string(index=False))
    print("\nPredicted class counts:")
    print(pd.Series(pred).value_counts().to_string())
    if "true_label" in out.columns:
        acc = (out["true_label"] == out["predicted_class"]).mean()
        print(f"\nAgreement with annotations: {acc:.3f}")
        print("(If this record was part of the training set, the value is optimistic.)")
    print(f"\nFull results saved to {args.out}")


if __name__ == "__main__":
    main()
