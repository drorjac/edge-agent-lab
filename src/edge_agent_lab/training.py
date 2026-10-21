"""Train/validation-only model selection; held-out test evaluation is a separate step."""

from __future__ import annotations

import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .common import environment, read_json, seed_everything, sha256, write_json
from .data import Split, corrupt, ensure_data, load_split, normalization, simulate
from .metrics import benchmark, classification, predict, subject_bootstrap
from .models import SensorNet, load_model, save_model


def train_model(
    model: SensorNet,
    train: Split,
    val: Split,
    out: Path,
    *,
    seed: int,
    epochs: int,
    teacher=None,
    augment: bool = False,
    learning_rate: float = 0.001,
    resume_weights: Path | None = None,
) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    if resume_weights:
        model.load_state_dict(load_model(resume_weights).state_dict())
    # Statistics belong only to this run's training split, including fine-tuning.
    mean, scale = normalization(train.x)
    model.mean.copy_(torch.from_numpy(mean))
    model.scale.copy_(torch.from_numpy(scale))
    if teacher:
        teacher.eval()
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y)),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    best, history, best_state = -1.0, [], None
    started = time.monotonic()
    for epoch in range(epochs):
        model.train()
        loss_sum = 0.0
        for x, y in loader:
            if augment:
                # Rotation, sensor gain, and small additive noise in raw units.
                x = x * (1 + 0.12 * torch.randn(len(x), 9, 1)) + 0.015 * torch.randn_like(x)
                if torch.rand(()) < 0.15:
                    x[:, 3:6] = 0
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            if teacher:
                with torch.no_grad():
                    target = teacher(x)
                temperature = 3.0
                kd = F.kl_div(
                    F.log_softmax(logits / temperature, dim=1),
                    F.softmax(target / temperature, dim=1),
                    reduction="batchmean",
                )
                loss = 0.5 * loss + 0.5 * temperature**2 * kd
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(y)
        scheduler.step()
        metrics = classification(val.y, predict(model, val.x))
        row = {
            "epoch": epoch + 1,
            "loss": loss_sum / len(train.y),
            "val_accuracy": metrics["accuracy"],
            "val_macro_f1": metrics["macro_f1"],
        }
        history.append(row)
        if metrics["macro_f1"] > best:
            best = metrics["macro_f1"]
            best_state = copy.deepcopy(model.state_dict())
        if epoch == 0 or (epoch + 1) % 5 == 0:
            print(
                f"{out.name}: epoch {epoch + 1}/{epochs} loss={row['loss']:.4f} "
                f"val_f1={row['val_macro_f1']:.4f}",
                flush=True,
            )
        write_json(out / "history.json", history)
    model.load_state_dict(best_state)
    model.eval()
    torch.save({"config": model.config(), "state": best_state}, out / "model.pt")
    result = {
        "name": out.name,
        "seed": seed,
        "epochs": epochs,
        "augment": augment,
        "distillation": teacher is not None,
        "parameters": sum(p.numel() for p in model.parameters()),
        "train_rows": len(train.y),
        "train_source": "procedural" if (train.subjects == -1).all() else "real",
        "validation": classification(val.y, predict(model, val.x)),
        "benchmark": benchmark(model, val.x),
        "elapsed_seconds": time.monotonic() - started,
        "checkpoint_sha256": sha256(out / "model.pt"),
        "environment": environment(),
    }
    write_json(out / "result.json", result)
    return result


def run_edge(
    data_root: Path, output: Path, seeds: list[int], epochs: int = 40, extended: bool = True
) -> dict:
    manifest = ensure_data(data_root)
    output.mkdir(parents=True, exist_ok=True)
    spec = {"epochs": epochs, "dataset_sha256": manifest["archive_sha256"], "protocol": 1}
    spec_path = output / "study_spec.json"
    if spec_path.exists() and read_json(spec_path) != spec:
        raise ValueError("Study protocol changed; choose a new output directory")
    for result_path in output.glob("seed-*/*/result.json"):
        cached = read_json(result_path)
        expected = (
            max(10, epochs // 2) if result_path.parent.name.startswith("synthetic_") else epochs
        )
        if "epochs" in cached and cached["epochs"] != expected:
            raise ValueError("Cached epoch budget differs; choose a new output directory")
    write_json(spec_path, spec)
    train, val = load_split(data_root, "train"), load_split(data_root, "val")
    results = []
    for seed in seeds:
        seed_everything(seed)
        base = output / f"seed-{seed}"
        jobs = [
            ("teacher", {"width": 32, "fusion": True}, False, False),
            ("student", {"width": 12}, False, False),
            ("distilled", {"width": 12}, True, False),
            ("robust_distilled", {"width": 12}, True, True),
        ]
        if extended:
            jobs += [("accel_only", {"width": 12, "accel_only": True}, False, False)]
        for name, config, distilled, augmented in jobs:
            folder = base / name
            if (folder / "result.json").exists():
                result = read_json(folder / "result.json")
            else:
                seed_everything(seed)
                teacher = load_model(base / "teacher/model.pt") if distilled else None
                result = train_model(
                    SensorNet(**config),
                    train,
                    val,
                    folder,
                    seed=seed,
                    epochs=epochs,
                    teacher=teacher,
                    augment=augmented,
                )
            results.append(result)
        if extended:
            for randomized in (False, True):
                name = "synthetic_randomized" if randomized else "synthetic_fixed"
                folder = base / name
                synthetic = simulate(len(train.y), seed, randomized)
                if not (folder / "result.json").exists():
                    seed_everything(seed)
                    train_model(
                        SensorNet(width=12),
                        synthetic,
                        val,
                        folder,
                        seed=seed,
                        epochs=max(10, epochs // 2),
                    )
                results.append(read_json(folder / "result.json"))
            # Compare scratch vs pretraining using the same small labeled real-data subset.
            rng = np.random.default_rng(seed)
            idx = np.concatenate(
                [rng.choice(np.flatnonzero(train.y == k), 80, replace=False) for k in range(6)]
            )
            limited = Split(train.x[idx], train.y[idx], train.subjects[idx])
            for name, resume in [
                ("fewshot_scratch", None),
                ("sim_to_real", base / "synthetic_randomized/model.pt"),
            ]:
                folder = base / name
                if not (folder / "result.json").exists():
                    seed_everything(seed)
                    train_model(
                        SensorNet(width=12),
                        limited,
                        val,
                        folder,
                        seed=seed,
                        epochs=epochs,
                        resume_weights=resume,
                    )
                results.append(read_json(folder / "result.json"))
        # Real integer kernels for Linear layers; convolution remains floating point.
        name = "dynamic_int8"
        folder = base / name
        if not (folder / "result.json").exists():
            folder.mkdir(parents=True, exist_ok=True)
            model = load_model(base / "distilled/model.pt")
            torch.backends.quantized.engine = "qnnpack"
            quantized = torch.ao.quantization.quantize_dynamic(
                model, {nn.Linear}, dtype=torch.qint8
            )
            save_model(quantized, folder / "model.pt", model.config())
            result = {
                "name": name,
                "seed": seed,
                "quantization": "dynamic int8 Linear only; float Conv1d",
                "validation": classification(val.y, predict(quantized, val.x)),
                "benchmark": benchmark(quantized, val.x),
                "environment": environment(),
                "checkpoint_sha256": sha256(folder / "model.pt"),
            }
            write_json(folder / "result.json", result)
        results.append(read_json(folder / "result.json"))
    summary = {
        "dataset_sha256": manifest["archive_sha256"],
        "seeds": seeds,
        "selection_split": "validation",
        "results": results,
    }
    write_json(output / "summary.json", summary)
    return summary


def evaluate_test(data_root: Path, output: Path, layout: str = "edge") -> dict:
    """Read-only final evaluation; these metrics never enter candidate selection."""
    test = load_split(data_root, "test")
    results = []
    if layout not in {"edge", "search"}:
        raise ValueError("Evaluation layout must be edge or search")
    pattern = "seed-*/*/model.pt" if layout == "edge" else "trained/*/model.pt"
    for path in sorted(output.glob(pattern)):
        name = path.parent.name
        quantized = name == "dynamic_int8"
        if quantized:
            torch.backends.quantized.engine = "qnnpack"
        model = load_model(path, quantized=quantized)
        probs = predict(model, test.x)
        row = {
            "candidate": str(path.parent.relative_to(output)),
            "clean": classification(test.y, probs),
            "uncertainty": subject_bootstrap(test.y, probs, test.subjects),
            "stress": {
                kind: classification(test.y, predict(model, corrupt(test.x, kind, 17)))
                for kind in ("noise", "gain", "gyro_dropout", "rotation")
            },
        }
        results.append(row)
    report = {
        "split": "official subject-held-out test",
        "used_for_selection": False,
        "results": results,
    }
    write_json(output / "test_evaluation.json", report)
    return report
