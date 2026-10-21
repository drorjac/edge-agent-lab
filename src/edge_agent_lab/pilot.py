"""Budgeted planner -> experiment runner -> critic -> promotion workflow.

The controller is an explicit adaptive policy, not an LLM reasoning claim. It
consumes measured validation results and never reads test metrics to rank models.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field

from .common import environment, read_json, sha256, write_json
from .data import corrupt, ensure_data, load_split
from .metrics import benchmark, classification, predict
from .models import load_model
from .steward import Steward


class Objective(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    min_macro_f1: float = Field(default=0.85, ge=0, le=1)
    max_p95_ms: float = Field(default=2.0, gt=0)
    max_bytes: int = Field(default=150_000, gt=0)
    min_dropout_f1: float = Field(default=0.45, ge=0, le=1)
    max_trials: int = Field(default=8, ge=1, le=100)
    max_seconds: float = Field(default=600, gt=0)


def feasible(result: dict, objective: Objective) -> bool:
    return (
        result["validation"]["macro_f1"] >= objective.min_macro_f1
        and result["benchmark"]["p95_ms"] <= objective.max_p95_ms
        and result["benchmark"]["state_dict_bytes"] <= objective.max_bytes
        and result["dropout_macro_f1"] >= objective.min_dropout_f1
    )


def pareto_front(rows: list[dict]) -> list[str]:
    def dominates(a, b):
        av = [
            a["validation"]["macro_f1"],
            -a["benchmark"]["p95_ms"],
            -a["benchmark"]["state_dict_bytes"],
            a["dropout_macro_f1"],
        ]
        bv = [
            b["validation"]["macro_f1"],
            -b["benchmark"]["p95_ms"],
            -b["benchmark"]["state_dict_bytes"],
            b["dropout_macro_f1"],
        ]
        return all(x >= y for x, y in zip(av, bv, strict=False)) and any(
            x > y for x, y in zip(av, bv, strict=False)
        )

    return [
        row["candidate"]
        for row in rows
        if not any(dominates(other, row) for other in rows if other is not row)
    ]


def rank_candidates(rows: list[dict], objective: Objective) -> list[dict]:
    accepted = [row for row in rows if feasible(row, objective)]
    return sorted(
        accepted,
        key=lambda r: (
            0.7 * r["validation"]["macro_f1"] + 0.3 * r["dropout_macro_f1"],
            -r["benchmark"]["p95_ms"],
        ),
        reverse=True,
    )


def run_pilot(
    data_root: Path, edge_root: Path, output: Path, objective: Objective, seed: int = 7
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    manifest = ensure_data(data_root)
    preflight = Steward(data_root, output / "steward").run(
        "Check provenance subject leakage and missing gyroscope robustness before deployment"
    )
    if preflight["status"] != "complete":
        raise RuntimeError("Steward preflight did not complete")
    actual_audits = [e["observation"] for e in preflight["trace"] if e["action"]["tool"] == "audit"]
    if not all(row["healthy"] for row in actual_audits):
        raise ValueError("Dataset quality preflight failed")
    val = load_split(data_root, "val")
    paths = sorted((edge_root / f"seed-{seed}").glob("*/result.json"))
    if not paths:
        raise ValueError("Train edge models before running the pilot")
    queue = []
    for path in paths:
        result = read_json(path)
        # Prior scores are train/validation experiment metadata, never test metrics.
        score = result["validation"]["macro_f1"]
        estimated_size = result["benchmark"]["state_dict_bytes"]
        score -= max(0, estimated_size / objective.max_bytes - 1) * 0.1
        queue.append((score, path))
    queue.sort(key=lambda item: item[0], reverse=True)
    started, measured, trace = time.monotonic(), [], []
    while queue and len(measured) < objective.max_trials:
        if time.monotonic() - started > objective.max_seconds:
            break
        # After a failed dropout gate, prioritize robust training over clean-score exploitation.
        if measured and not any(feasible(row, objective) for row in measured):
            queue.sort(key=lambda item: ("robust" in item[1].parent.name, item[0]), reverse=True)
        _, path = queue.pop(0)
        name = path.parent.name
        checkpoint = path.parent / "model.pt"
        if name == "dynamic_int8":
            torch.backends.quantized.engine = "qnnpack"
        model = load_model(checkpoint, quantized=name == "dynamic_int8")
        clean = classification(val.y, predict(model, val.x))
        dropout = classification(val.y, predict(model, corrupt(val.x, "gyro_dropout", seed)))
        row = {
            "candidate": name,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "validation": clean,
            "dropout_macro_f1": dropout["macro_f1"],
            "benchmark": benchmark(model, val.x),
        }
        row["feasible"] = feasible(row, objective)
        measured.append(row)
        trace.append(
            {
                "step": len(measured),
                "actor": "planner",
                "action": "evaluate_candidate",
                "candidate": name,
                "reason": "Prioritize robust candidate after gate failure"
                if len(measured) > 1 and "robust" in name
                else "Validation quality and size prior",
                "critic": {
                    "accepted": row["feasible"],
                    "f1_margin": clean["macro_f1"] - objective.min_macro_f1,
                    "latency_margin_ms": objective.max_p95_ms - row["benchmark"]["p95_ms"],
                    "size_margin_bytes": objective.max_bytes - row["benchmark"]["state_dict_bytes"],
                    "dropout_margin": dropout["macro_f1"] - objective.min_dropout_f1,
                },
            }
        )
        write_json(
            output / "checkpoint.json",
            {
                "objective": objective.model_dump(),
                "measured": measured,
                "trace": trace,
                "dataset_sha256": manifest["archive_sha256"],
            },
        )
    ranked = rank_candidates(measured, objective)
    winner = ranked[0] if ranked else None
    report = {
        "status": "selected" if winner else "no_feasible_candidate",
        "objective": objective.model_dump(),
        "selection_split": "validation",
        "test_metrics_used": False,
        "dataset_sha256": manifest["archive_sha256"],
        "preflight_run_id": preflight["run_id"],
        "trials": len(measured),
        "elapsed_seconds": time.monotonic() - started,
        "winner": winner,
        "pareto_front": pareto_front(measured),
        "candidates": measured,
        "trace": trace,
        "environment": environment(),
        "budget_semantics": "Wall-time checked between indivisible candidate evaluations.",
        "deployment": "Candidate selected for review; no production deployment performed.",
    }
    write_json(output / "selection.json", report)
    return report


def stream_monitor(data_root: Path, selection_path: Path, output: Path) -> dict:
    """Labeled validation replay, explicitly synthetic outage, no online-learning claim."""
    selection = read_json(selection_path)
    winner = selection.get("winner")
    if not winner:
        result = {"status": "skipped", "reason": "No feasible candidate was selected"}
        write_json(output / "telemetry.json", result)
        return result
    val = load_split(data_root, "val")
    quantized = winner["candidate"] == "dynamic_int8"
    if quantized:
        torch.backends.quantized.engine = "qnnpack"
    model = load_model(Path(winner["checkpoint"]), quantized=quantized)
    rng = np.random.default_rng(101)
    indices = rng.permutation(len(val.y))
    events = []
    reference = val.x.mean((0, 2))
    scale = val.x.std((0, 2)).clip(0.01)
    for step, idx in enumerate(np.array_split(indices, 12)):
        outage = step >= 6
        x = corrupt(val.x[idx], "gyro_dropout") if outage else val.x[idx]
        stats = classification(val.y[idx], predict(model, x))
        gyro_variance = float(x[:, 3:6].var())
        events.append(
            {
                "window": step,
                "rows": len(idx),
                "synthetic_outage": outage,
                "accuracy": stats["accuracy"],
                "macro_f1": stats["macro_f1"],
                "mean_shift_score": float(np.max(abs(x.mean((0, 2)) - reference) / scale)),
                "gyro_variance": gyro_variance,
                "alert": gyro_variance < 1e-8,
                "action": "request sensor inspection; keep candidate in shadow evaluation"
                if gyro_variance < 1e-8
                else "observe",
            }
        )
    report = {
        "source": "shuffled labeled validation replay with synthetic gyro outage halfway",
        "candidate": winner["candidate"],
        "events": events,
        "false_alerts": sum(e["alert"] and not e["synthetic_outage"] for e in events),
        "outage_detection_rate": float(
            np.mean([e["alert"] for e in events if e["synthetic_outage"]])
        ),
    }
    write_json(output / "telemetry.json", report)
    return report
