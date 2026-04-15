"""
data_loader.py
==============
Handles loading of MERL Shopping Dataset .mat files and
simulated data generation for demo/GitHub mode.

MERL Dataset: 106 videos, 41 subjects, 5 shopping actions
Actions: Reach to Shelf | Retract from Shelf | Hand in Shelf | Inspect Product | Inspect Shelf
FPS: 15 frames per second
"""

import os
import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #

ACTION_LABELS = {
    1: "Reach to Shelf",
    2: "Retract from Shelf",
    3: "Hand in Shelf",
    4: "Inspect Product",
    5: "Inspect Shelf",
}

ACTION_COLORS = {
    "Reach to Shelf":     "#00C9A7",
    "Retract from Shelf": "#845EC2",
    "Hand in Shelf":      "#FF6F91",
    "Inspect Product":    "#FFC75F",
    "Inspect Shelf":      "#4D8EFF",
}

FPS = 15  # MERL dataset frame rate


# --------------------------------------------------------------------------- #
# Real Data Loader
# --------------------------------------------------------------------------- #

def load_frame_names(results_path: str = "Results") -> list:
    """Return list of frame filename strings from frame_names.txt."""
    path = os.path.join(results_path, "frame_names.txt")
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_detected_actions(results_path: str = "Results") -> pd.DataFrame:
    """
    Load DetectedActions/1.mat … 5.mat and combine into a single DataFrame.
    Each .mat row = [start_frame, end_frame, confidence_score].
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("Run: pip install scipy")

    records = []
    detected_path = os.path.join(results_path, "DetectedActions")

    for action_id in range(1, 6):
        mat_path = os.path.join(detected_path, f"{action_id}.mat")
        if not os.path.exists(mat_path):
            continue
        try:
            mat = scipy.io.loadmat(mat_path)
            data_keys = [k for k in mat.keys() if not k.startswith("__")]
            if not data_keys:
                continue
            data = mat[data_keys[0]]
            for row in data:
                start, end, score = int(row[0]), int(row[1]), float(row[2])
                records.append({
                    "action_id":    action_id,
                    "action":       ACTION_LABELS[action_id],
                    "start_frame":  start,
                    "end_frame":    end,
                    "score":        round(score, 4),
                    "start_sec":    round(start / FPS, 2),
                    "end_sec":      round(end / FPS, 2),
                    "duration_sec": round((end - start) / FPS, 2),
                })
        except Exception as e:
            print(f"  Warning: Could not load action {action_id}: {e}")

    return pd.DataFrame(records)


def load_label_data(labels_path: str = "Labels_MERL_Shopping_Dataset") -> pd.DataFrame:
    """
    Load per-subject ground-truth label .mat files.
    Each file xx_yy_label.mat contains tlabs cell array with action intervals.
    """
    try:
        import scipy.io
    except ImportError:
        raise ImportError("Run: pip install scipy")

    records = []
    if not os.path.exists(labels_path):
        return pd.DataFrame()

    for fname in os.listdir(labels_path):
        if not fname.endswith("_label.mat"):
            continue
        parts = fname.replace("_label.mat", "").split("_")
        if len(parts) < 2:
            continue
        try:
            subject_id, session_id = int(parts[0]), int(parts[1])
        except ValueError:
            continue

        mat_path = os.path.join(labels_path, fname)
        try:
            mat = scipy.io.loadmat(mat_path, squeeze_me=True, struct_as_record=False)
            tlabs = mat.get("tlabs", None)
            if tlabs is None:
                continue
            for action_id in range(1, 6):
                intervals = tlabs[action_id - 1]
                if intervals is None or (hasattr(intervals, "__len__") and len(intervals) == 0):
                    continue
                intervals = np.atleast_2d(intervals)
                for interval in intervals:
                    start, end = int(interval[0]), int(interval[1])
                    records.append({
                        "subject_id":   subject_id,
                        "session_id":   session_id,
                        "action_id":    action_id,
                        "action":       ACTION_LABELS[action_id],
                        "start_frame":  start,
                        "end_frame":    end,
                        "start_sec":    round(start / FPS, 2),
                        "end_sec":      round(end / FPS, 2),
                        "duration_sec": round((end - start) / FPS, 2),
                        "score":        1.0,  # ground truth = full confidence
                    })
        except Exception as e:
            print(f"  Warning: {fname}: {e}")

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Simulated Data Generator
# --------------------------------------------------------------------------- #

def simulate_data(n_subjects: int = 41, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic simulated data mirroring MERL structure.
    Each subject has up to 3 sessions × ~2 min video at 15 fps.
    Action frequency and duration params are calibrated to match
    the real dataset's reported mAP and per-frame accuracy.
    """
    rng = np.random.default_rng(seed)
    total_frames = 1800  # 2 min × 15 fps

    action_params = {
        1: {"mean_dur": 28,  "std_dur": 10, "mean_freq": 9,  "std_freq": 2},
        2: {"mean_dur": 22,  "std_dur": 8,  "mean_freq": 9,  "std_freq": 2},
        3: {"mean_dur": 65,  "std_dur": 22, "mean_freq": 5,  "std_freq": 2},
        4: {"mean_dur": 95,  "std_dur": 30, "mean_freq": 6,  "std_freq": 2},
        5: {"mean_dur": 48,  "std_dur": 16, "mean_freq": 7,  "std_freq": 2},
    }

    records = []
    for subject_id in range(1, n_subjects + 1):
        # Each subject appears in 1-3 sessions
        n_sessions = rng.integers(1, 4)
        for session_id in range(1, n_sessions + 1):
            for action_id, p in action_params.items():
                n_occ = max(1, int(rng.normal(p["mean_freq"], p["std_freq"])))
                for _ in range(n_occ):
                    dur = max(5, int(rng.normal(p["mean_dur"], p["std_dur"])))
                    start = int(rng.integers(0, max(1, total_frames - dur - 10)))
                    end = min(start + dur, total_frames)
                    score = float(rng.uniform(0.45, 0.99))
                    records.append({
                        "subject_id":   subject_id,
                        "session_id":   session_id,
                        "action_id":    action_id,
                        "action":       ACTION_LABELS[action_id],
                        "start_frame":  start,
                        "end_frame":    end,
                        "score":        round(score, 4),
                        "start_sec":    round(start / FPS, 2),
                        "end_sec":      round(end / FPS, 2),
                        "duration_sec": round((end - start) / FPS, 2),
                    })

    return pd.DataFrame(records)


# --------------------------------------------------------------------------- #
# Smart Loader — tries real data first, falls back to simulated
# --------------------------------------------------------------------------- #

def get_data(
    results_path: str = "Results",
    labels_path: str = "Labels_MERL_Shopping_Dataset",
    force_simulate: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    Returns (DataFrame, source_label).
    Tries real data first; falls back to simulated with a clear label.
    """
    if force_simulate:
        return simulate_data(), "simulated"

    df = load_detected_actions(results_path)
    if not df.empty:
        return df, "real_detected"

    df = load_label_data(labels_path)
    if not df.empty:
        return df, "real_labels"

    return simulate_data(), "simulated"

