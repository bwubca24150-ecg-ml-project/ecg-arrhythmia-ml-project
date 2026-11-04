"""
data_loader.py
--------------
Finds and reads MIT-BIH Arrhythmia Database records.

The loader searches the data/ folder recursively, so it works whether the
files are directly inside data/ or inside a nested folder such as
data/mitdb-1.0.0/mit-bih-arrhythmia-database-1.0.0/
"""
import os
from pathlib import Path

import numpy as np
import wfdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ECG_DATA_DIR", PROJECT_ROOT / "data"))

# Records 102, 104, 107, 217 contain paced beats. The AAMI standard
# recommends excluding them from arrhythmia classification experiments.
PACED_RECORDS = {"102", "104", "107", "217"}

CLASSES = ["N", "S", "V", "F", "Q"]

CLASS_DESCRIPTIONS = {
    "N": "Normal heartbeat group",
    "S": "Supraventricular ectopic beat",
    "V": "Ventricular ectopic beat",
    "F": "Fusion beat",
    "Q": "Other / unknown beat",
}

# MIT-BIH beat symbol -> AAMI class
AAMI_MAP = {
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    "A": "S", "a": "S", "J": "S", "S": "S",
    "V": "V", "E": "V",
    "F": "F",
    "/": "Q", "f": "Q", "Q": "Q",
}


def find_records(data_dir=DATA_DIR, exclude_paced=True):
    """Return {record_name: path_without_extension} for all usable records."""
    data_dir = Path(data_dir)
    records = {}
    if not data_dir.exists():
        return records
    for hea in sorted(data_dir.rglob("*.hea")):
        name = hea.stem
        if not name.isdigit():
            continue
        if exclude_paced and name in PACED_RECORDS:
            continue
        base = hea.with_suffix("")
        if not (base.with_suffix(".dat").exists() and base.with_suffix(".atr").exists()):
            continue
        records.setdefault(name, base)
    return dict(sorted(records.items()))


def load_record(record_path):
    """
    Read one record.

    Returns
    -------
    signal  : 1-D numpy array (MLII lead if available, otherwise channel 0)
    fs      : sampling frequency (Hz)
    samples : sample index of every annotation
    symbols : annotation symbol of every annotation
    """
    record_path = str(record_path)
    record = wfdb.rdrecord(record_path)
    ann = wfdb.rdann(record_path, "atr")

    names = [n.strip().upper() for n in record.sig_name]
    channel = names.index("MLII") if "MLII" in names else 0
    signal = record.p_signal[:, channel].astype(float)
    return signal, float(record.fs), np.asarray(ann.sample), np.asarray(ann.symbol)


def beat_labels(samples, symbols):
    """Keep only beat annotations and convert them to the 5 AAMI classes."""
    keep = np.array([s in AAMI_MAP for s in symbols], dtype=bool)
    samples = np.asarray(samples)[keep]
    labels = np.array([AAMI_MAP[s] for s in np.asarray(symbols)[keep]])
    return samples, labels
