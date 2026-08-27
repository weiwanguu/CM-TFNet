"""Dataset: sliding-window next-frame bimanual Fz (lift start to full water exit)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from . import config as cfg
from .medium_state import (
    build_medium_state_for_trial,
    finger_force_max,
    load_object_geometry,
)


FEATURE_DIM = 15  # 7L + 7R + s
FZ_L_IDX = 2
FZ_R_IDX = 9


def apply_fz_scale(X: np.ndarray, Y: np.ndarray, fz_scale: float) -> tuple[np.ndarray, np.ndarray]:
    """Scale only left/right |Fz| channels and labels."""
    if abs(float(fz_scale) - 1.0) < 1e-12:
        return X, Y
    X = X.copy()
    Y = Y.copy()
    X[:, FZ_L_IDX] *= fz_scale
    X[:, FZ_R_IDX] *= fz_scale
    Y *= fz_scale
    return X, Y


def _norm_proximity(p: np.ndarray) -> np.ndarray:
    """Proximity in the CSV is already /65535 into [0,1]; read as-is."""
    return np.asarray(p, dtype=np.float64)


@dataclass
class TrialMeta:
    folder: str
    file: str
    object_name: str
    t_peak: float
    t0: float
    H_m: float
    d_m: float
    n: int
    i_lift: int
    i_exit: int


def lift_exit_indices(
    time_s: np.ndarray,
    s: np.ndarray,
    t0: float,
    exit_extra_s: float | None = None,
) -> tuple[int, int]:
    """
    Lift-start frame -> prediction end frame.
    Default end is the first time s reaches 1 (just fully out of water);
    if exit_extra_s>0, extend further by that many seconds.
    """
    t = np.asarray(time_s, dtype=np.float64)
    i_lift = int(np.searchsorted(t, t0, side="left"))
    i_lift = min(max(i_lift, 0), len(t) - 1)
    air = np.where(np.asarray(s) >= 1.0 - 1e-6)[0]
    if len(air):
        i_air = int(air[0])
    else:
        i_air = len(t) - 1
    extra = float(cfg.EXIT_EXTRA_S if exit_extra_s is None else exit_extra_s)
    if extra > 0:
        t_end = float(t[i_air]) + extra
        i_exit = int(np.searchsorted(t, t_end, side="left"))
        i_exit = min(i_exit, len(t) - 1)
    else:
        i_exit = i_air
    if i_exit < i_lift:
        i_exit = i_lift
    return i_lift, i_exit


def load_trial_arrays(
    csv_path: Path,
    H_m: float,
    d_m: float,
    exit_extra_s: float | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, TrialMeta]:
    """
    Returns:
      X: (T, 15) raw features (unnormalized)
      Y: (T, 2)  current-frame Fz_L, Fz_R
      s: (T,)
      time_s: (T,)
      meta
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    t = df["time_s"].to_numpy(dtype=np.float64)
    # Two force sensors per finger: pick the larger |xyz| per axis
    Lf = finger_force_max(df, "L")
    Rf = finger_force_max(df, "R")
    Lp = _norm_proximity(df["L_proximity"].to_numpy(dtype=np.float64))
    Rp = _norm_proximity(df["R_proximity"].to_numpy(dtype=np.float64))
    La = df[["L_roll", "L_pitch", "L_yaw"]].to_numpy(dtype=np.float64)
    Ra = df[["R_roll", "R_pitch", "R_yaw"]].to_numpy(dtype=np.float64)

    s, t_peak, t0 = build_medium_state_for_trial(df, H_m, d_m)
    i_lift, i_exit = lift_exit_indices(t, s, t0, exit_extra_s=exit_extra_s)

    left = np.concatenate([Lf, Lp[:, None], La], axis=1)  # (T,7)
    right = np.concatenate([Rf, Rp[:, None], Ra], axis=1)
    X = np.concatenate([left, right, s[:, None]], axis=1).astype(np.float32)
    Y = np.stack([np.abs(Lf[:, 2]), np.abs(Rf[:, 2])], axis=1).astype(np.float32)

    folder = csv_path.parent.name
    meta = TrialMeta(
        folder=folder,
        file=csv_path.name,
        object_name=folder,
        t_peak=float(t_peak),
        t0=float(t0),
        H_m=float(H_m),
        d_m=float(d_m),
        n=len(df),
        i_lift=i_lift,
        i_exit=i_exit,
    )
    return X, Y, s.astype(np.float32), t.astype(np.float64), meta


def discover_trials(data_root: Path | None = None) -> list[tuple[Path, dict]]:
    root = Path(data_root or cfg.DATA_ROOT)
    geom = load_object_geometry()
    trials = []
    for folder, g in sorted(geom.items()):
        # Use grasp_logs_* under DATA_ROOT only; skip csv backups
        d = root / folder
        if not d.is_dir():
            continue
        if "backup" in str(d).lower():
            continue
        for csv in sorted(d.glob("*.csv"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem):
            trials.append((csv, g))
    return trials


def split_train_val_test(
    trials: list[tuple[Path, dict]],
    val_objects=cfg.VAL_OBJECTS,
    test_objects=cfg.TEST_OBJECTS,
) -> tuple[list, list, list]:
    """Object-level 6:2:2 split: specified val/test objects, the rest for train."""
    val_set = set(val_objects)
    test_set = set(test_objects)
    train, val, test = [], [], []
    for item in trials:
        name = item[0].parent.name
        if name in test_set:
            test.append(item)
        elif name in val_set:
            val.append(item)
        else:
            train.append(item)
    return train, val, test


def split_train_val_test_by_trial(
    trials: list[tuple[Path, dict]],
    seed: int = cfg.SEED,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
) -> tuple[list, list, list, dict]:
    """
    Random 6:2:2 split of trials per object; every object appears in train/val/test.
    For 15 trials: 9 / 3 / 3.
    Returns train, val, test, split_detail.
    """
    rng = np.random.default_rng(seed)
    by_obj: dict[str, list[tuple[Path, dict]]] = {}
    for item in trials:
        by_obj.setdefault(item[0].parent.name, []).append(item)

    train, val, test = [], [], []
    split_detail: dict[str, dict[str, list[str]]] = {}

    for obj, items in sorted(by_obj.items()):
        items = sorted(items, key=lambda it: int(it[0].stem) if it[0].stem.isdigit() else it[0].stem)
        n = len(items)
        # 15 -> 9/3/3; in general n*ratios, leftover goes to train
        n_train = int(round(n * ratios[0]))
        n_val = int(round(n * ratios[1]))
        n_test = n - n_train - n_val
        if n >= 3:
            n_val = max(n_val, 1)
            n_test = max(n_test, 1)
            n_train = n - n_val - n_test
            n_train = max(n_train, 1)

        idx = np.arange(n)
        rng.shuffle(idx)
        i_train = idx[:n_train]
        i_val = idx[n_train : n_train + n_val]
        i_test = idx[n_train + n_val :]

        tr = [items[i] for i in sorted(i_train)]
        va = [items[i] for i in sorted(i_val)]
        te = [items[i] for i in sorted(i_test)]
        train.extend(tr)
        val.extend(va)
        test.extend(te)
        split_detail[obj] = {
            "train": [p.name for p, _ in tr],
            "val": [p.name for p, _ in va],
            "test": [p.name for p, _ in te],
        }

    return train, val, test, split_detail


def trials_from_split_detail(
    trials: list[tuple[Path, dict]],
    split_detail: dict,
    which: str,
) -> list[tuple[Path, dict]]:
    """Rebuild a trial list for one split from split_detail."""
    want = {}
    for obj, parts in split_detail.items():
        want[obj] = set(parts[which])
    out = []
    for item in trials:
        obj = item[0].parent.name
        if obj in want and item[0].name in want[obj]:
            out.append(item)
    return out


def compute_norm_stats(
    trials: list[tuple[Path, dict]],
    fz_scale: float = 1.0,
) -> dict:
    """Estimate stats from lift-to-exit frames only."""
    xs = []
    ys = []
    for csv, g in trials:
        X, Y, s, t, meta = load_trial_arrays(csv, g["H_m"], g["d_m"])
        X, Y = apply_fz_scale(X, Y, fz_scale)
        seg = slice(meta.i_lift, meta.i_exit + 1)
        xs.append(X[seg])
        ys.append(Y[seg])
    Xall = np.concatenate(xs, axis=0)
    Yall = np.concatenate(ys, axis=0)
    return {
        "x_mean": Xall.mean(axis=0).tolist(),
        "x_std": (Xall.std(axis=0) + 1e-6).tolist(),
        "y_mean": Yall.mean(axis=0).tolist(),
        "y_std": (Yall.std(axis=0) + 1e-6).tolist(),
        "fz_scale": float(fz_scale),
    }


class GraspUnrollDataset(Dataset):
    """
    Multi-step unroll samples: normalized context on [t-W, t+K) and K-step Fz labels.
    Training writes predicted Fz back into the context via scheduled sampling.
    """

    def __init__(
        self,
        trials: list[tuple[Path, dict]],
        stats: dict,
        window: int = cfg.WINDOW,
        unroll_steps: int = cfg.UNROLL_STEPS,
        stride: int = 4,
        fz_scale: float = 1.0,
    ):
        self.window = window
        self.unroll_steps = int(unroll_steps)
        self.stride = int(stride)
        self.fz_scale = float(fz_scale)
        self.x_mean = np.asarray(stats["x_mean"], dtype=np.float32)
        self.x_std = np.asarray(stats["x_std"], dtype=np.float32)

        self.samples: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        self.metas: list[TrialMeta] = []

        K = self.unroll_steps
        for csv, g in trials:
            X, Y, s, t, meta = load_trial_arrays(csv, g["H_m"], g["d_m"])
            X, Y = apply_fz_scale(X, Y, self.fz_scale)
            Xn = (X - self.x_mean) / self.x_std
            i0, i1 = meta.i_lift, meta.i_exit
            # Need a full K-step span inside the supervised segment, plus W frames of history
            last_start = i1 - K + 1
            if last_start < i0:
                continue
            for tgt_start in range(i0, last_start + 1, self.stride):
                if tgt_start < window:
                    continue
                x_ctx = Xn[tgt_start - window : tgt_start + K].astype(np.float32)  # (W+K, 15)
                y_seq = Y[tgt_start : tgt_start + K].astype(np.float32)  # (K, 2)
                w_seq = np.array(
                    [
                        1.0
                        + (cfg.CROSS_MEDIUM_LOSS_WEIGHT - 1.0)
                        * float((s[tgt_start + k] > 0.0) and (s[tgt_start + k] < 1.0))
                        for k in range(K)
                    ],
                    dtype=np.float32,
                )
                self.samples.append((x_ctx, y_seq, w_seq))
            self.metas.append(meta)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        x, y, w = self.samples[idx]
        return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(w)



def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
