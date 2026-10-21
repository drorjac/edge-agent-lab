"""Classification, calibration, subject bootstrap, and batch-one timing."""

from __future__ import annotations

import io
import time

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification(y: np.ndarray, probabilities: np.ndarray) -> dict:
    pred = probabilities.argmax(1)
    confidence = probabilities.max(1)
    correct = pred == y
    ece = 0.0
    for low, high in zip(np.linspace(0, 1, 11)[:-1], np.linspace(0, 1, 11)[1:], strict=False):
        mask = (confidence > low) & (confidence <= high)
        if mask.any():
            ece += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", labels=range(6), zero_division=0)),
        "ece_10bin": ece,
        "confusion_matrix": confusion_matrix(y, pred, labels=range(6)).tolist(),
    }


@torch.inference_mode()
def predict(model, x: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    return torch.cat(
        [model(torch.from_numpy(x[i : i + batch])).softmax(1) for i in range(0, len(x), batch)]
    ).numpy()


@torch.inference_mode()
def benchmark(model, x: np.ndarray, repeats: int = 200, warmup: int = 30) -> dict:
    model.eval()
    tensor = torch.from_numpy(x[:1])
    for _ in range(warmup):
        model(tensor)
    durations = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model(tensor)
        durations.append((time.perf_counter_ns() - start) / 1e6)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    return {
        "p50_ms": float(np.median(durations)),
        "p95_ms": float(np.percentile(durations, 95)),
        "repeats": repeats,
        "warmup": warmup,
        "batch_size": 1,
        "state_dict_bytes": stream.tell(),
        "backend": "pytorch_cpu",
        "threads": torch.get_num_threads(),
    }


def subject_bootstrap(y, probabilities, subjects, seed: int = 0, repeats: int = 500) -> dict:
    rng = np.random.default_rng(seed)
    unique = np.unique(subjects)
    groups = {s: np.flatnonzero(subjects == s) for s in unique}
    values = []
    for _ in range(repeats):
        idx = np.concatenate([groups[s] for s in rng.choice(unique, len(unique), replace=True)])
        values.append(float((probabilities[idx].argmax(1) == y[idx]).mean()))
    return {
        "accuracy_95ci": np.percentile(values, [2.5, 97.5]).tolist(),
        "unit": "subject",
        "replicates": repeats,
    }
