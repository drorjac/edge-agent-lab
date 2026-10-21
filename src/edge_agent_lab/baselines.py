"""Simple statistical baselines contextualize whether neural complexity is useful."""

from __future__ import annotations

import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .common import environment, write_json
from .data import corrupt, ensure_data, load_split
from .metrics import classification, subject_bootstrap


def features(x: np.ndarray) -> np.ndarray:
    spectrum = abs(np.fft.rfft(x, axis=2))
    spectrum[:, :, 0] = 0  # dominant motion frequency excludes gravity/DC
    values = [
        x.mean(2),
        x.std(2),
        x.min(2),
        x.max(2),
        np.quantile(x, 0.75, axis=2) - np.quantile(x, 0.25, axis=2),
        np.square(x).mean(2),
        spectrum.argmax(2) * 50 / 128,
        np.square(np.diff(x, axis=2)).mean(2),
    ]
    return np.concatenate(values, axis=1).astype("float32")


def run_baselines(data_root: Path, output: Path) -> dict:
    ensure_data(data_root)
    output.mkdir(parents=True, exist_ok=True)
    train, val, test = [load_split(data_root, name) for name in ("train", "val", "test")]
    x_train, x_val, x_test = map(lambda split: features(split.x), (train, val, test))
    classifiers = {
        "statistical_logistic": make_pipeline(
            StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, random_state=7)
        ),
        "statistical_boosting": HistGradientBoostingClassifier(
            max_iter=150, max_leaf_nodes=15, l2_regularization=1, random_state=7
        ),
    }
    rows = []
    with threadpool_limits(limits=2):
        for name, model in classifiers.items():
            started = time.monotonic()
            model.fit(x_train, train.y)
            elapsed = time.monotonic() - started
            val_probs, test_probs = model.predict_proba(x_val), model.predict_proba(x_test)
            durations = []
            for step in range(230):
                before = time.perf_counter_ns()
                model.predict_proba(features(val.x[:1]))
                if step >= 30:
                    durations.append((time.perf_counter_ns() - before) / 1e6)
            path = output / f"{name}.joblib"
            joblib.dump(model, path)
            row = {
                "name": name,
                "validation": classification(val.y, val_probs),
                "test": classification(test.y, test_probs),
                "test_uncertainty": subject_bootstrap(test.y, test_probs, test.subjects),
                "stress": {
                    kind: classification(
                        test.y, model.predict_proba(features(corrupt(test.x, kind, 17)))
                    )
                    for kind in ("noise", "gain", "gyro_dropout", "rotation")
                },
                "train_seconds": elapsed,
                "features": x_train.shape[1],
                "benchmark": {
                    "p50_ms": float(np.median(durations)),
                    "p95_ms": float(np.percentile(durations, 95)),
                    "model_file_bytes": path.stat().st_size,
                    "scope": "CPU batch one; includes feature extraction",
                    "threads": 2,
                },
                "selection": "Fixed baseline parameters; no test tuning",
                "environment": environment(),
            }
            rows.append(row)
            print(
                f"{name}: val_f1={row['validation']['macro_f1']:.4f} "
                f"test_f1={row['test']['macro_f1']:.4f}",
                flush=True,
            )
            write_json(output / "results.json", {"results": rows})
    return {"results": rows}
