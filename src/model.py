"""
model.py
--------
Random Forest model: build, split, evaluate, save, load, predict.
"""
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from .data_loader import PROJECT_ROOT, CLASSES
from .features import extract_features, feature_names
from .preprocessing import DEFAULT_RR

MODEL_PATH = PROJECT_ROOT / "models" / "rf_model.joblib"


def build_model(n_estimators=200, random_state=42):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced_subsample",  # rare classes (S, F, Q) get more weight
        n_jobs=-1,
        random_state=random_state,
    )


def split_by_record(X, y, groups, test_size=0.25, random_state=42):
    """
    GroupShuffleSplit: all beats from one record go either to training or to
    testing, never both. This avoids data leakage between train and test.
    """
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(gss.split(X, y, groups))
    return train_idx, test_idx


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    labels = CLASSES
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "report_text": classification_report(
            y_test, y_pred, labels=labels, zero_division=0
        ),
        "report_dict": classification_report(
            y_test, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=labels),
        "labels": labels,
    }


def save_model(model, rr_defaults, path=MODEL_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "feature_names": feature_names(), "rr_defaults": rr_defaults},
        path,
    )


def load_model(path=MODEL_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\nRun 'python train.py' first."
        )
    return joblib.load(path)


def predict_beats(bundle, beats, rr=None):
    """
    Predict classes for beat windows.
    If rr is None, the average RR values seen during training are used.
    Returns (labels, probabilities).
    """
    beats = np.asarray(beats, dtype=float)
    if rr is None:
        defaults = np.asarray(bundle.get("rr_defaults", [DEFAULT_RR] * 3))
        rr = np.tile(defaults, (len(beats), 1))
    X = extract_features(beats, rr)
    model = bundle["model"]
    return model.predict(X), model.predict_proba(X)
