"""
preprocessing.py
----------------
ECG cleaning (bandpass filter), beat segmentation and RR-interval calculation.
"""
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

from .data_loader import load_record, beat_labels

# Beat window around the R peak (samples at 360 Hz): 90 before + 110 after
PRE = 90
POST = 110
BEAT_LEN = PRE + POST  # 200

RR_COLUMNS = ["pre_rr", "post_rr", "avg_rr"]
DEFAULT_RR = 0.8  # seconds, used only when no RR information exists


def bandpass_filter(signal, fs, low=0.5, high=40.0, order=3):
    """Butterworth bandpass filter (zero phase) to remove baseline wander and noise."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signal)


def compute_rr_features(r_peaks, fs):
    """
    For every beat return [pre_rr, post_rr, avg_rr] in seconds.
      pre_rr  : interval from previous beat
      post_rr : interval to next beat
      avg_rr  : average of the last 10 pre-RR intervals
    """
    r = np.asarray(r_peaks, dtype=float)
    n = len(r)
    if n < 2:
        return np.full((n, 3), DEFAULT_RR)
    rr = np.diff(r) / fs
    pre = np.concatenate([[rr[0]], rr])
    post = np.concatenate([rr, [rr[-1]]])
    avg = pd.Series(pre).rolling(10, min_periods=1).mean().to_numpy()
    return np.column_stack([pre, post, avg])


def segment_beats(signal, r_peaks, pre=PRE, post=POST):
    """Cut a fixed window around every R peak. Returns (beats, valid_mask)."""
    r = np.asarray(r_peaks, dtype=int)
    valid = (r >= pre) & (r + post <= len(signal))
    if not valid.any():
        return np.empty((0, pre + post)), valid
    beats = np.stack([signal[p - pre:p + post] for p in r[valid]])
    return beats, valid


def build_record_dataset(record_path):
    """
    Full pipeline for one record:
    load -> filter -> keep beats -> RR features -> segment.

    Returns beats (n, 200), rr (n, 3), labels (n,)
    """
    signal, fs, samples, symbols = load_record(record_path)
    signal = bandpass_filter(signal, fs)
    r_peaks, labels = beat_labels(samples, symbols)
    rr = compute_rr_features(r_peaks, fs)
    beats, valid = segment_beats(signal, r_peaks)
    return beats, rr[valid], labels[valid]


def build_dataset(records, verbose=True):
    """Run build_record_dataset() on many records and stack the results."""
    all_beats, all_rr, all_labels, all_groups = [], [], [], []
    for i, (name, path) in enumerate(records.items(), 1):
        beats, rr, labels = build_record_dataset(path)
        all_beats.append(beats)
        all_rr.append(rr)
        all_labels.append(labels)
        all_groups.append(np.full(len(labels), name))
        if verbose:
            print(f"  [{i}/{len(records)}] record {name}: {len(labels)} beats")
    return (
        np.vstack(all_beats),
        np.vstack(all_rr),
        np.concatenate(all_labels),
        np.concatenate(all_groups),
    )
