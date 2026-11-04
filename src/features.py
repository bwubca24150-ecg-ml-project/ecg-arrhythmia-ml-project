"""
features.py
-----------
Turns every heartbeat (200 samples) into a numeric feature vector.

Feature groups
  1. Statistical / morphology features of the beat
  2. Down-sampled, normalised beat shape
  3. RR-interval (timing) features
"""
import numpy as np
from scipy.stats import skew, kurtosis

from .preprocessing import PRE, BEAT_LEN

DOWNSAMPLE = 4
_EPS = 1e-8

_BASE_NAMES = [
    "mean", "std", "max", "min", "peak_to_peak", "argmax_pos", "argmin_pos",
    "energy", "skewness", "kurtosis", "r_amplitude", "qrs_energy", "width_proxy",
]
_SHAPE_NAMES = [f"shape_{i}" for i in range(0, BEAT_LEN, DOWNSAMPLE)]
_RR_NAMES = ["pre_rr", "post_rr", "avg_rr", "pre_over_avg", "post_over_avg", "pre_over_post"]


def feature_names():
    return _BASE_NAMES + _SHAPE_NAMES + _RR_NAMES


def extract_features(beats, rr):
    """
    beats : (n, 200) filtered beat windows
    rr    : (n, 3)   columns pre_rr, post_rr, avg_rr (seconds)
    returns (n, n_features) float array
    """
    beats = np.asarray(beats, dtype=float)
    rr = np.asarray(rr, dtype=float)

    mean = beats.mean(axis=1)
    std = beats.std(axis=1)
    mx = beats.max(axis=1)
    mn = beats.min(axis=1)
    ptp = mx - mn
    argmax = beats.argmax(axis=1) / BEAT_LEN
    argmin = beats.argmin(axis=1) / BEAT_LEN
    energy = (beats ** 2).sum(axis=1)
    sk = skew(beats, axis=1)
    ku = kurtosis(beats, axis=1)
    r_amp = beats[:, PRE]
    qrs_energy = (beats[:, PRE - 15:PRE + 15] ** 2).sum(axis=1)

    centred = beats - mean[:, None]
    peak = np.abs(centred).max(axis=1)
    width = (np.abs(centred) > 0.5 * peak[:, None]).mean(axis=1)

    base = np.column_stack(
        [mean, std, mx, mn, ptp, argmax, argmin, energy, sk, ku, r_amp, qrs_energy, width]
    )

    shape = (centred / (std[:, None] + _EPS))[:, ::DOWNSAMPLE]

    pre, post, avg = rr[:, 0], rr[:, 1], rr[:, 2]
    rr_feats = np.column_stack(
        [pre, post, avg, pre / (avg + _EPS), post / (avg + _EPS), pre / (post + _EPS)]
    )

    return np.nan_to_num(np.hstack([base, shape, rr_feats]), nan=0.0, posinf=0.0, neginf=0.0)
