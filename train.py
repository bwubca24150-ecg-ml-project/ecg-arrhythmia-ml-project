"""
train.py
--------
Trains the Random Forest ECG classifier on the MIT-BIH Arrhythmia Database.

Usage:
    python train.py                    # all records
    python train.py --max_records 10   # quick test run
"""
import argparse
import json
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loader import DATA_DIR, PROJECT_ROOT, CLASSES, find_records
from src.preprocessing import build_dataset
from src.features import extract_features, feature_names
from src.model import build_model, split_by_record, evaluate_model, save_model

REPORT_DIR = PROJECT_ROOT / "reports"


def plot_confusion_matrix(cm, labels, path):
    cm = np.asarray(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, data, title, fmt in [
        (axes[0], cm, "Confusion Matrix (counts)", "d"),
        (axes[1], cm_norm, "Confusion Matrix (row-normalised)", ".2f"),
    ]:
        im = ax.imshow(data, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        threshold = data.max() / 2 if data.max() > 0 else 0.5
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if data[i, j] > threshold else "black")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_feature_importance(model, path, top=15):
    names = np.array(feature_names())
    imp = model.feature_importances_
    idx = np.argsort(imp)[::-1][:top][::-1]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(names[idx], imp[idx], color="#2b6cb0")
    ax.set_title(f"Top {top} Feature Importances (Random Forest)")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Train ECG arrhythmia classifier")
    parser.add_argument("--max_records", type=int, default=None, help="use only first N records (quick test)")
    parser.add_argument("--test_size", type=float, default=0.25, help="fraction of records used for testing")
    parser.add_argument("--trees", type=int, default=200, help="number of trees in the Random Forest")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    records = find_records()
    if not records:
        print(f"No MIT-BIH records found inside: {DATA_DIR}")
        print("Put the extracted dataset (files like 100.dat, 100.hea, 100.atr) in the data/ folder.")
        return
    if args.max_records:
        records = dict(list(records.items())[: args.max_records])

    print(f"Found {len(records)} records. Loading, filtering and segmenting beats...")
    t0 = time.time()
    beats, rr, y, groups = build_dataset(records)
    print(f"Total beats: {len(y)}  ({time.time() - t0:.1f}s)")

    unique, counts = np.unique(y, return_counts=True)
    print("Class distribution:", dict(zip(unique.tolist(), counts.tolist())))

    print("Extracting features...")
    X = extract_features(beats, rr)
    print(f"Feature matrix: {X.shape}")

    train_idx, test_idx = split_by_record(X, y, groups, args.test_size, args.seed)
    print(f"Train beats: {len(train_idx)} | Test beats: {len(test_idx)}")
    print(f"Test records: {sorted(set(groups[test_idx].tolist()))}")

    print("Training Random Forest...")
    t0 = time.time()
    model = build_model(args.trees, args.seed)
    model.fit(X[train_idx], y[train_idx])
    print(f"Training finished in {time.time() - t0:.1f}s")

    result = evaluate_model(model, X[test_idx], y[test_idx])

    print("\n=== Classification Report ===")
    print(result["report_text"])
    print(f"Accuracy: {result['accuracy']:.4f}")
    print("Confusion matrix (rows = true, columns = predicted):")
    print(result["labels"])
    print(result["confusion_matrix"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "classification_report.txt").write_text(
        result["report_text"] + f"\nAccuracy: {result['accuracy']:.4f}\n"
    )
    plot_confusion_matrix(result["confusion_matrix"], result["labels"], REPORT_DIR / "confusion_matrix.png")
    plot_feature_importance(model, REPORT_DIR / "feature_importance.png")
    metrics = {
        "accuracy": result["accuracy"],
        "n_records": len(records),
        "n_train_beats": int(len(train_idx)),
        "n_test_beats": int(len(test_idx)),
        "test_records": sorted(set(groups[test_idx].tolist())),
        "report": result["report_dict"],
        "confusion_matrix": result["confusion_matrix"].tolist(),
        "labels": result["labels"],
    }
    (REPORT_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    rr_defaults = np.median(rr[train_idx], axis=0).tolist()
    save_model(model, rr_defaults)
    print(f"\nModel saved to models/rf_model.joblib")
    print(f"Reports saved to {REPORT_DIR}")


if __name__ == "__main__":
    main()
