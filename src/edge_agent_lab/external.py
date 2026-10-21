"""An explicitly mismatched, independent HAR70+ transfer probe.

This is not a comparable HAR70+ benchmark: thigh channels are unused, gyroscope
is unavailable, body acceleration is approximated, and label coverage is reduced.
"""

from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import torch
from scipy.signal import butter, sosfiltfilt

from .common import read_json, sha256, write_json
from .data import LABELS
from .metrics import classification, predict, subject_bootstrap
from .models import load_model

URL = "https://archive.ics.uci.edu/static/public/780/har70.zip"
EXPECTED_SHA256 = "5db89b85c1a4f494f51fcc6661eb2ad41dd7efbc4799c776e794c56fb439e5c3"
MAPPING = {1: 0, 4: 1, 5: 2, 7: 3, 6: 4, 8: 5}


def bridge_recording(total: np.ndarray, labels: np.ndarray, timestamps: np.ndarray) -> tuple:
    if total.shape != (len(labels), 3) or len(timestamps) != len(labels):
        raise ValueError("Invalid external recording shape")
    if not np.isfinite(total).all():
        raise ValueError("Nonfinite external sensor values")
    delta = np.diff(timestamps).astype("timedelta64[ms]").astype("int64")
    breaks = np.flatnonzero((delta < 10) | (delta > 30)) + 1
    boundaries = np.r_[0, breaks, len(total)]
    windows, targets = [], []
    counts = {
        "raw_samples": len(total),
        "time_discontinuities": len(breaks),
        "candidate_windows": 0,
        "transition_or_unmapped_windows": 0,
        "retained_windows": 0,
    }
    lowpass = butter(3, 0.3, fs=50, output="sos")
    for begin, end in zip(boundaries[:-1], boundaries[1:], strict=True):
        if end - begin < 128:
            continue
        acceleration = total[begin:end]
        gravity = sosfiltfilt(lowpass, acceleration, axis=0)
        body = acceleration - gravity
        for start in range(0, len(acceleration) - 127, 128):
            counts["candidate_windows"] += 1
            activity = labels[begin + start : begin + start + 128]
            if activity[0] not in MAPPING or not (activity == activity[0]).all():
                counts["transition_or_unmapped_windows"] += 1
                continue
            window = np.concatenate(
                [
                    body[start : start + 128].T,
                    np.zeros((3, 128)),
                    acceleration[start : start + 128].T,
                ],
                axis=0,
            )
            windows.append(window.astype("float32"))
            targets.append(MAPPING[int(activity[0])])
    counts["retained_windows"] = len(targets)
    array = np.stack(windows) if windows else np.empty((0, 9, 128), dtype="float32")
    return array, np.array(targets, dtype="int64"), counts


def prepare_external(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "har70.zip"
    partial = archive.with_suffix(".zip.part")
    if not archive.exists():
        if not partial.exists() or not zipfile.is_zipfile(partial):
            with urllib.request.urlopen(URL, timeout=180) as response, partial.open("wb") as stream:
                while block := response.read(1024 * 1024):
                    stream.write(block)
        partial.replace(archive)
    digest = sha256(archive)
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise ValueError("HAR70+ archive checksum mismatch")
    all_x, all_y, all_subjects, audit = [], [], [], []
    with zipfile.ZipFile(archive) as zf:
        for name in sorted(n for n in zf.namelist() if n.endswith(".csv")):
            content = zf.read(name)
            matrix = np.loadtxt(
                io.BytesIO(content), delimiter=",", skiprows=1, usecols=(1, 2, 3, 7)
            )
            timestamps = np.loadtxt(
                io.BytesIO(content), delimiter=",", skiprows=1, usecols=(0,), dtype="datetime64[ms]"
            )
            x, y, counts = bridge_recording(matrix[:, :3], matrix[:, 3].astype("int64"), timestamps)
            subject = int(Path(name).stem)
            all_x.append(x)
            all_y.append(y)
            all_subjects.append(np.full(len(y), subject, dtype="int64"))
            audit.append({"subject": subject, **counts})
            print(f"HAR70+ subject {subject}: {len(y)} retained windows", flush=True)
    x, y, subjects = np.concatenate(all_x), np.concatenate(all_y), np.concatenate(all_subjects)
    np.savez_compressed(root / "probe.npz", x=x, y=y, subjects=subjects)
    manifest = {
        "dataset": "HAR70+",
        "doi": "10.24432/C5CW3D",
        "url": URL,
        "archive_sha256": digest,
        "license": "CC BY 4.0",
        "rows": len(y),
        "subjects": sorted(map(int, np.unique(subjects))),
        "labels": LABELS,
        "class_counts": np.bincount(y, minlength=6).tolist(),
        "subject_audit": audit,
        "mapping": MAPPING,
        "probe_sha256": sha256(root / "probe.npz"),
        "policy": "Non-overlapping 128-sample windows; homogeneous mapped label; no time gaps",
        "bridge": "Back-only; 0.3Hz noncausal gravity estimate; absent gyro is zero",
        "orientation_variants": ["as_recorded", "fixed_180_degree_y_rotation"],
        "selection_use": "none; external descriptive evaluation only",
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def evaluate_external(root: Path, edge_root: Path, output: Path, seed: int = 7) -> dict:
    manifest_path = root / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else prepare_external(root)
    if sha256(root / "probe.npz") != manifest["probe_sha256"]:
        raise ValueError("External probe integrity mismatch")
    with np.load(root / "probe.npz", allow_pickle=False) as arrays:
        x, y, subjects = arrays["x"], arrays["y"], arrays["subjects"]
    rows = []
    for name in ("teacher", "student", "distilled", "robust_distilled", "dynamic_int8"):
        checkpoint = edge_root / f"seed-{seed}" / name / "model.pt"
        if not checkpoint.exists():
            continue
        if name == "dynamic_int8":
            torch.backends.quantized.engine = "qnnpack"
        model = load_model(checkpoint, quantized=name == "dynamic_int8")
        for orientation in ("as_recorded", "fixed_180_degree_y_rotation"):
            signals = x.copy()
            if orientation != "as_recorded":
                signals[:, [0, 2, 6, 8]] *= -1
            probs = predict(model, signals)
            rows.append(
                {
                    "candidate": name,
                    "orientation": orientation,
                    "metrics": classification(y, probs),
                    "uncertainty": subject_bootstrap(y, probs, subjects),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
    report = {
        "dataset": manifest,
        "results": rows,
        "limits": "Mismatched population, placement, preprocessing, absent gyro; descriptive only.",
        "selection_use": False,
    }
    write_json(output / "results.json", report)
    return report
