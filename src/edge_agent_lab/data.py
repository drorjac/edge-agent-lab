"""Versioned UCI ingestion, subject splits, and sensor simulation.

Never split overlapping windows at random: adjacent windows from the same person
would leak identity and recording context into validation.
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from filelock import FileLock

from .common import sha256, write_json

URL = (
    "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
)
# Filled from the initial verified download and enforced on subsequent ingestion.
EXPECTED_SHA256 = "c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031"
CHANNELS = [
    f"{sensor}_{axis}" for sensor in ("body_acc", "body_gyro", "total_acc") for axis in "xyz"
]
LABELS = ["walking", "walking_upstairs", "walking_downstairs", "sitting", "standing", "laying"]
VALIDATION_SUBJECTS = {5, 15, 19, 26}


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    subjects: np.ndarray


def download(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "uci_har.zip"
    if not archive.exists():
        partial = archive.with_suffix(".part")
        request = urllib.request.Request(URL, headers={"User-Agent": "EdgeAgentLab/0.1"})
        try:
            with (
                urllib.request.urlopen(request, timeout=120) as response,
                partial.open("wb") as out,
            ):
                while block := response.read(1024 * 1024):
                    out.write(block)
            if not zipfile.is_zipfile(partial):
                raise ValueError("Dataset response is not a ZIP archive")
            partial.replace(archive)
        finally:
            partial.unlink(missing_ok=True)
    digest = sha256(archive)
    if EXPECTED_SHA256 and digest != EXPECTED_SHA256:
        raise ValueError(f"Dataset checksum mismatch: {digest}")
    return archive


def _dataset_zip(archive: Path) -> zipfile.ZipFile:
    outer = zipfile.ZipFile(archive)
    if any(name.endswith("train/y_train.txt") for name in outer.namelist()):
        return outer
    member = next(name for name in outer.namelist() if name.endswith("UCI HAR Dataset.zip"))
    nested = io.BytesIO(outer.read(member))
    outer.close()
    return zipfile.ZipFile(nested)


def prepare(root: Path) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / ".prepare.lock"), timeout=300):
        return _prepare_unlocked(root)


def _prepare_unlocked(root: Path) -> dict:
    archive = download(root)
    with _dataset_zip(archive) as zf:
        names = zf.namelist()

        def load(suffix: str) -> np.ndarray:
            matches = [name for name in names if name.endswith(suffix) and "__MACOSX" not in name]
            if len(matches) != 1:
                raise ValueError(f"Expected one dataset member for {suffix}: {matches}")
            return np.loadtxt(io.BytesIO(zf.read(matches[0])))

        splits = {}
        for part in ("train", "test"):
            x = np.stack(
                [load(f"{part}/Inertial Signals/{ch}_{part}.txt") for ch in CHANNELS], axis=1
            ).astype("float32")
            y = load(f"{part}/y_{part}.txt").astype("int64") - 1
            subjects = load(f"{part}/subject_{part}.txt").astype("int64")
            if not np.isfinite(x).all() or x.shape != (len(y), 9, 128):
                raise ValueError("Malformed sensor dataset")
            splits[part] = Split(x, y, subjects)
    train = splits["train"]
    mask = np.isin(train.subjects, sorted(VALIDATION_SUBJECTS))
    if not mask.any() or mask.all():
        raise ValueError("Validation subjects must select a proper training subset")
    splits["val"] = Split(train.x[mask], train.y[mask], train.subjects[mask])
    splits["train"] = Split(train.x[~mask], train.y[~mask], train.subjects[~mask])
    assert_disjoint(splits)
    for name, split in splits.items():
        np.savez_compressed(root / f"{name}.npz", x=split.x, y=split.y, subjects=split.subjects)
    manifest = {
        "dataset": "UCI HAR",
        "doi": "10.24432/C54S4K",
        "url": URL,
        "license": "CC BY 4.0",
        "archive_sha256": sha256(archive),
        "channels": CHANNELS,
        "sample_rate_hz": 50,
        "window_samples": 128,
        "labels": LABELS,
        "split_policy": "subject-disjoint; official test unchanged",
        "splits": {
            name: {
                "rows": len(s.y),
                "subjects": sorted(map(int, set(s.subjects))),
                "class_counts": np.bincount(s.y, minlength=6).tolist(),
                "sha256": sha256(root / f"{name}.npz"),
            }
            for name, s in splits.items()
        },
    }
    write_json(root / "manifest.json", manifest)
    return manifest


def assert_disjoint(splits: dict[str, Split]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if set(splits[a].subjects) & set(splits[b].subjects):
                raise ValueError(f"Subject leakage between {a} and {b}")


def load_split(root: Path, name: str) -> Split:
    with np.load(root / f"{name}.npz", allow_pickle=False) as data:
        return Split(data["x"], data["y"], data["subjects"])


def ensure_data(root: Path) -> dict:
    if not (root / "manifest.json").exists():
        return prepare(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["archive_sha256"] != EXPECTED_SHA256:
        raise ValueError("Cached manifest source identity does not match the pinned dataset")
    for name, meta in manifest["splits"].items():
        if sha256(root / f"{name}.npz") != meta["sha256"]:
            raise ValueError(f"Cached {name} data failed integrity check; run data prepare")
    return manifest


def normalization(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return x.mean(axis=(0, 2), keepdims=True), x.std(axis=(0, 2), keepdims=True).clip(1e-4)


def corrupt(x: np.ndarray, kind: str, seed: int = 0, severity: float = 1.0) -> np.ndarray:
    """Synthetic stress tests on real signals, not new measured datasets."""
    rng = np.random.default_rng(seed)
    out = x.copy()
    if kind == "noise":
        out += rng.normal(0, 0.06 * severity, out.shape).astype("float32")
    elif kind == "gain":
        out *= rng.uniform(
            max(0.1, 1 - 0.35 * severity), 1 + 0.35 * severity, (len(x), 9, 1)
        ).astype("float32")
    elif kind == "gyro_dropout":
        out[:, 3:6] = 0
    elif kind == "rotation":
        angle = rng.uniform(-0.5, 0.5, len(x)) * severity
        c, s = np.cos(angle)[:, None], np.sin(angle)[:, None]
        for start in (0, 3, 6):
            a, b = out[:, start].copy(), out[:, start + 1].copy()
            out[:, start], out[:, start + 1] = c * a - s * b, s * a + c * b
    elif kind != "clean":
        raise ValueError(f"Unknown corruption: {kind}")
    return out


def simulate(n: int, seed: int = 0, randomized: bool = True) -> Split:
    """Procedural motion prior: periodic gait, gravity, and low-motion postures.

    Parameters are hand-authored, never estimated from validation/test signals.
    It is deliberately approximate: synthetic-only transfer is an experiment,
    not a claim of a validated biomechanical simulator.
    """
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 6
    rng.shuffle(y)
    x = np.zeros((n, 9, 128), dtype="float32")
    t = np.arange(128) / 50
    for i, label in enumerate(y):
        phase = rng.uniform(0, 2 * np.pi, 3)
        frequency = [1.8, 1.55, 2.1, 0.15, 0.1, 0.08][label]
        amplitude = [0.22, 0.29, 0.34, 0.008, 0.005, 0.004][label]
        if randomized:
            frequency *= rng.uniform(0.65, 1.4)
            amplitude *= rng.uniform(0.55, 1.6)
        wave = np.sin(2 * np.pi * frequency * t + phase[:, None])
        harmonic = np.sin(4 * np.pi * frequency * t + phase[:, None])
        body = amplitude * (wave + 0.25 * harmonic)
        gyro = amplitude * 2 * np.cos(2 * np.pi * frequency * t + phase[:, None])
        gravity = np.array([0.95, 0.15, 0.1]) if label != 5 else np.array([0.1, 0.1, 0.98])
        if label == 3:
            gravity = np.array([0.8, 0.35, 0.15])
        x[i, :3] = body
        x[i, 3:6] = gyro
        x[i, 6:] = body + gravity[:, None]
    x += rng.normal(0, 0.015 if randomized else 0.003, x.shape).astype("float32")
    if randomized:
        x = corrupt(x, "rotation", seed + 1, severity=0.5)
    return Split(x, y.astype("int64"), np.full(n, -1, dtype="int64"))
