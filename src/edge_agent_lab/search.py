"""A bounded agent workflow that launches real training experiments on demand."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .common import environment, read_json, seed_everything, sha256, write_json
from .data import corrupt, ensure_data, load_split
from .metrics import benchmark, classification, predict
from .models import SensorNet, load_model
from .pilot import Objective, feasible, pareto_front, rank_candidates
from .steward import Steward
from .training import train_model


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    width: Literal[4, 6, 8, 12, 16]
    augment: bool
    epochs: int = Field(ge=1, le=80)
    seed: int = Field(ge=0)

    @property
    def name(self):
        return f"width-{self.width}-" + ("robust" if self.augment else "base")


def propose_experiments(objective: Objective, epochs: int, seed: int, sample) -> list[Experiment]:
    """Estimate storage structurally; measure accuracy only by running the experiment."""
    fitting = []
    for width in (4, 6, 8, 12, 16):
        seed_everything(seed)
        estimate = benchmark(SensorNet(width=width).eval(), sample, repeats=1, warmup=0)
        if estimate["state_dict_bytes"] <= objective.max_bytes:
            fitting.append(width)
    plans = []
    for width in reversed(fitting):
        # Robust augmentation is a measured experimental treatment, not assumed better.
        for augment in (True, False):
            plans.append(Experiment(width=width, augment=augment, epochs=epochs, seed=seed))
    return plans


def run_search(
    data_root: Path,
    teacher_path: Path,
    output: Path,
    objective: Objective,
    epochs: int = 30,
    seed: int = 7,
) -> dict:
    seed_everything(seed)
    manifest = ensure_data(data_root)
    output.mkdir(parents=True, exist_ok=True)
    preflight = Steward(data_root, output / "steward").run(
        "Audit subject leakage provenance and deployment policy"
    )
    audits = [r["observation"] for r in preflight["trace"] if r["action"]["tool"] == "audit"]
    if preflight["status"] != "complete" or not audits or not all(r["healthy"] for r in audits):
        raise ValueError("Data preflight failed")
    train, val = load_split(data_root, "train"), load_split(data_root, "val")
    plans = propose_experiments(objective, epochs, seed, val.x[:1])
    spec = {
        "objective": objective.model_dump(),
        "epochs": epochs,
        "seed": seed,
        "dataset_sha256": manifest["archive_sha256"],
        "teacher_sha256": sha256(teacher_path),
    }
    spec_path = output / "study_spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError("Search configuration changed; select a fresh output directory")
    write_json(spec_path, spec)
    teacher = load_model(teacher_path)
    measured, trace = [], []
    started = time.monotonic()
    while plans and len(measured) < objective.max_trials:
        if time.monotonic() - started >= objective.max_seconds:
            break
        proposal = plans.pop(0)
        folder = output / "trained" / proposal.name
        reused = (folder / "result.json").exists()
        if not reused:
            seed_everything(seed)
            train_model(
                SensorNet(width=proposal.width),
                train,
                val,
                folder,
                seed=seed,
                epochs=proposal.epochs,
                teacher=teacher,
                augment=proposal.augment,
            )
        model = load_model(folder / "model.pt")
        training_result = read_json(folder / "result.json")
        row = {
            "training_seconds": training_result["elapsed_seconds"],
            "training_epochs": proposal.epochs,
            "candidate": proposal.name,
            "checkpoint": str(folder / "model.pt"),
            "checkpoint_sha256": sha256(folder / "model.pt"),
            "validation": classification(val.y, predict(model, val.x)),
            "dropout_macro_f1": classification(
                val.y, predict(model, corrupt(val.x, "gyro_dropout"))
            )["macro_f1"],
            "benchmark": benchmark(model, val.x),
            "proposal": proposal.model_dump(),
            "reused_training": reused,
        }
        row["feasible"] = feasible(row, objective)
        measured.append(row)
        trace.append(
            {
                "step": len(measured),
                "planner": "Largest structurally fitting width; paired robustness treatment",
                "action": "train_distilled_candidate",
                "proposal": proposal.model_dump(),
                "runner": {"trained": not reused, "checkpoint_sha256": row["checkpoint_sha256"]},
                "critic": {
                    "feasible": row["feasible"],
                    "f1_margin": row["validation"]["macro_f1"] - objective.min_macro_f1,
                    "size_margin": objective.max_bytes - row["benchmark"]["state_dict_bytes"],
                    "dropout_margin": row["dropout_macro_f1"] - objective.min_dropout_f1,
                },
            }
        )
        # Prioritize robust proposals after a dropout-gate failure.
        if row["dropout_macro_f1"] < objective.min_dropout_f1:
            plans.sort(key=lambda p: p.augment, reverse=True)
        ranked = rank_candidates(measured, objective)
        report = {
            "status": "selected" if ranked else "no_feasible_candidate",
            "objective": objective.model_dump(),
            "winner": ranked[0] if ranked else None,
            "trials": len(measured),
            "candidates": measured,
            "trace": trace,
            "selection_split": "validation",
            "test_metrics_used": False,
            "pareto_front": pareto_front(measured),
            "spec": spec,
            "elapsed_seconds": time.monotonic() - started,
            "environment": environment(),
            "budget_semantics": "Checked between jobs; one job may exceed the remaining time.",
        }
        write_json(output / "selection.json", report)
    if not measured:
        report = {
            "status": "no_feasible_candidate",
            "winner": None,
            "trials": 0,
            "objective": objective.model_dump(),
            "candidates": [],
            "trace": [],
            "reason": "No untrained architecture fits the storage budget or time expired",
            "selection_split": "validation",
            "test_metrics_used": False,
            "spec": spec,
        }
        write_json(output / "selection.json", report)
    return report
