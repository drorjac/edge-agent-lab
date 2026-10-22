"""CPU INT8 deployment comparisons for the pinned PyTorch 2.2 sensor models."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.ao.nn.intrinsic.qat import freeze_bn_stats
from torch.ao.quantization import (
    disable_observer,
    get_default_qat_qconfig_mapping,
    get_default_qconfig_mapping,
)
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, prepare_qat_fx
from torch.utils.data import DataLoader, TensorDataset

from .common import environment, seed_everything, sha256, write_json
from .data import corrupt, ensure_data, load_split
from .metrics import classification, predict
from .models import load_model


def set_backend(backend: str) -> None:
    if backend == "none" or backend not in torch.backends.quantized.supported_engines:
        raise ValueError(f"Quantized backend {backend!r} is unavailable in this PyTorch build")
    torch.backends.quantized.engine = backend


def calibration_indices(rows: int, count: int, seed: int) -> np.ndarray:
    if rows < 1 or count < 1:
        raise ValueError("Calibration needs positive row and sample counts")
    return np.random.default_rng(seed).permutation(rows)[: min(count, rows)]


def prepare_static(model, calibration: np.ndarray, backend: str = "qnnpack"):
    """Observe representative raw training windows; no optimizer or labels are used."""
    set_backend(backend)
    if len(calibration) == 0:
        raise ValueError("Calibration cannot be empty")
    prepared = prepare_fx(
        copy.deepcopy(model).eval(),
        get_default_qconfig_mapping(backend),
        (torch.from_numpy(calibration[:1]),),
    )
    with torch.inference_mode():
        for start in range(0, len(calibration), 64):
            prepared(torch.from_numpy(calibration[start : start + 64]))
    return convert_fx(prepared).eval()


def layer_inventory(model) -> list[dict]:
    rows = []
    for name, module in model.named_modules():
        weight = getattr(module, "weight", None)
        if callable(weight):
            weight = weight()
        if isinstance(weight, torch.Tensor) and weight.ndim in (2, 3):
            rows.append(
                {
                    "name": name,
                    "module": type(module).__name__,
                    "weight_shape": list(weight.shape),
                    "weight_dtype": str(weight.dtype),
                    "quantized_weight": weight.is_quantized,
                    "weight_qscheme": str(weight.qscheme()) if weight.is_quantized else None,
                }
            )
    return rows


def assert_static_coverage(model) -> list[dict]:
    layers = layer_inventory(model)
    if len(layers) != 4 or not all(layer["quantized_weight"] for layer in layers):
        raise RuntimeError(f"Expected two INT8 convolution and two INT8 linear layers: {layers}")
    if not all(
        "quantized" in type(m).__module__
        for m in model.modules()
        if type(m).__name__ in {"ConvReLU1d", "Conv1d", "LinearReLU", "Linear"}
    ):
        raise RuntimeError("A compute layer remained floating point")
    return layers


def tune(model, train, val, epochs: int, seed: int, *, qat: bool) -> tuple[nn.Module, list]:
    """Select actual converted INT8 validation scores for QAT, float scores for control."""
    seed_everything(seed)
    initial = convert_fx(copy.deepcopy(model).eval()) if qat else copy.deepcopy(model).eval()
    best = classification(val.y, predict(initial, val.x))["macro_f1"]
    selected = initial
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.001)
    history = [{"epoch": 0, "validation_macro_f1": best}]
    for epoch in range(epochs):
        model.train()
        if qat and epoch >= 1:
            model.apply(freeze_bn_stats)
        if qat and epoch >= 2:
            model.apply(disable_observer)
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        candidate = convert_fx(copy.deepcopy(model).eval()) if qat else copy.deepcopy(model).eval()
        metrics = classification(val.y, predict(candidate, val.x))
        history.append({"epoch": epoch + 1, "validation_macro_f1": metrics["macro_f1"]})
        if metrics["macro_f1"] > best:
            selected, best = candidate, metrics["macro_f1"]
        print(
            "qat" if qat else "float control", epoch + 1, round(metrics["macro_f1"], 4), flush=True
        )
    return selected, history


def deployment_archive(model, example: np.ndarray, path: Path):
    """Export the inference graph and weights, then verify the loaded archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.from_numpy(example[:1])
    with torch.inference_mode():
        archive = torch.jit.trace(model.eval(), tensor)
        torch.jit.save(archive, str(path))
        restored = torch.jit.load(str(path), map_location="cpu").eval()
        np.testing.assert_allclose(
            restored(torch.from_numpy(example[:16])).numpy(),
            model(torch.from_numpy(example[:16])).numpy(),
            atol=1e-6,
            rtol=1e-5,
        )
    return restored


@torch.inference_mode()
def cpu_timing(model, x: np.ndarray, batch_size=1, repeats=100, warmup=30) -> dict:
    tensor = torch.from_numpy(x[:batch_size])
    for _ in range(warmup):
        model(tensor)
    times = []
    for _ in range(repeats):
        start = time.perf_counter_ns()
        model(tensor)
        times.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "p50_ms": float(np.median(times)),
        "p95_ms": float(np.percentile(times, 95)),
        "batch_size": len(tensor),
        "repeats": repeats,
        "warmup": warmup,
        "scope": "loaded TorchScript CPU forward; resident inputs; no loading/softmax/transfers",
    }


def run_study(
    checkpoint: Path,
    data_root: Path,
    output: Path,
    *,
    epochs=5,
    seed=7,
    backend="qnnpack",
    compact_checkpoint: Path | None = None,
) -> dict:
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    seed_everything(seed)
    set_backend(backend)
    manifest = ensure_data(data_root)
    train, val = load_split(data_root, "train"), load_split(data_root, "val")
    model = load_model(checkpoint)
    if model.fusion:
        raise ValueError("This comparison expects a sequential non-fusion student")
    output.mkdir(parents=True, exist_ok=True)
    idx = calibration_indices(len(train.x), 512, seed)
    calibration = train.x[idx]
    history = {}
    candidates = {"float_original": model}
    candidates["float_finetuned"], history["float_finetuned"] = tune(
        copy.deepcopy(model), train, val, epochs, seed, qat=False
    )
    candidates["dynamic_int8_linear"] = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model), {nn.Linear}, dtype=torch.qint8
    )
    for count in (64, 512):
        candidates[f"static_ptq_{count}"] = prepare_static(model, calibration[:count], backend)
    qat = prepare_qat_fx(
        copy.deepcopy(model).train(),
        get_default_qat_qconfig_mapping(backend),
        (torch.from_numpy(calibration[:1]),),
    )
    # Initialize observers from training-only windows without updating BN statistics.
    qat.eval()
    with torch.no_grad():
        for start in range(0, len(calibration), 64):
            qat(torch.from_numpy(calibration[start : start + 64]))
    candidates["qat_int8"], history["qat_int8"] = tune(qat, train, val, epochs, seed, qat=True)
    compact_identity = None
    if compact_checkpoint is not None:
        compact = load_model(compact_checkpoint)
        candidates["compact_float"] = compact
        candidates["compact_static_ptq_512"] = prepare_static(compact, calibration, backend)
        compact_identity = {"path": str(compact_checkpoint), "sha256": sha256(compact_checkpoint)}
    result = {
        "environment": environment(),
        "torch_quantization_backend": backend,
        "source_checkpoint": str(checkpoint),
        "source_sha256": sha256(checkpoint),
        "compact_checkpoint": compact_identity,
        "dataset_sha256": manifest["archive_sha256"],
        "seed": seed,
        "fine_tuning_epochs": epochs,
        "calibration": {
            "split": "train",
            "rows": len(idx),
            "indices": idx.tolist(),
            "class_counts": np.bincount(train.y[idx], minlength=6).tolist(),
            "subset64_class_counts": np.bincount(train.y[idx[:64]], minlength=6).tolist(),
            "split_sha256": sha256(data_root / "train.npz"),
            "normalization": "source training statistics embedded in checkpoint",
        },
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_rule": "smallest deployment archive within 0.02 validation F1 of original float",
        "candidates": [],
        "limits": (
            "One seed; CPU INT8 backend, float normalization/output and graph boundaries remain."
        ),
    }
    loaded = {}
    for name, candidate in candidates.items():
        layers = layer_inventory(candidate)
        if name.startswith("static_") or name in {"qat_int8", "compact_static_ptq_512"}:
            layers = assert_static_coverage(candidate)
        path = output / name / "model.ts"
        restored = deployment_archive(candidate, val.x, path)
        loaded[name] = restored
        row = {
            "name": name,
            "archive": str(path),
            "archive_sha256": sha256(path),
            "archive_bytes": path.stat().st_size,
            "layers": layers,
            "history": history.get(name, []),
            "archive_roundtrip_passed": True,
            "validation": classification(val.y, predict(restored, val.x)),
            "outage_validation": classification(
                val.y, predict(restored, corrupt(val.x, "gyro_dropout", seed=seed))
            ),
            "timing_rounds": [],
        }
        if name in history:
            # Epoch zero is eligible; an equal score never replaces the incumbent.
            improvements = [
                h
                for h in history[name]
                if h["validation_macro_f1"] == row["validation"]["macro_f1"]
            ]
            row["selected_finetuning_epoch"] = improvements[0]["epoch"] if improvements else 0
        if hasattr(candidate, "graph"):
            graph_path = path.parent / "graph.txt"
            graph_path.write_text(str(candidate.graph) + "\n")
        result["candidates"].append(row)
    # Benchmark the deployed representation, interleaving candidates after training.
    rng = np.random.default_rng(seed)
    for _ in range(3):
        for index in rng.permutation(len(result["candidates"])):
            row = result["candidates"][index]
            restored = loaded[row["name"]]
            row["timing_rounds"].append(
                {
                    "batch1": cpu_timing(restored, val.x),
                    "batch64": cpu_timing(restored, val.x, 64),
                }
            )
    for row in result["candidates"]:
        row["timing"] = {}
        for batch in ("batch1", "batch64"):
            samples = [t[batch] for t in row["timing_rounds"]]
            summary = dict(samples[0])
            for metric in ("p50_ms", "p95_ms"):
                summary[metric] = float(np.median([s[metric] for s in samples]))
            summary["p95_round_range_ms"] = [
                min(s["p95_ms"] for s in samples),
                max(s["p95_ms"] for s in samples),
            ]
            row["timing"][batch] = summary
    floor = result["candidates"][0]["validation"]["macro_f1"] - 0.02
    passing = [c for c in result["candidates"] if c["validation"]["macro_f1"] >= floor]
    winner = min(passing, key=lambda c: (c["archive_bytes"], -c["validation"]["macro_f1"]))
    result["selected"], result["minimum_validation_f1"] = winner["name"], floor
    test = load_split(data_root, "test")
    reference = predict(loaded["float_original"], test.x)
    for row in result["candidates"]:
        probs = predict(loaded[row["name"]], test.x)
        row["test"] = classification(test.y, probs)
        row["test_top1_agreement_with_original_float"] = float(
            (probs.argmax(1) == reference.argmax(1)).mean()
        )
        print(row["name"], round(row["validation"]["macro_f1"], 4), flush=True)
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/edge/seed-7/distilled/model.pt")
    )
    parser.add_argument("--compact-checkpoint", type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("data/har"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/quantization"))
    parser.add_argument("--report", type=Path, default=Path("reports/quantization.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--backend", default="qnnpack")
    args = parser.parse_args()
    result = run_study(
        args.checkpoint,
        args.data_root,
        args.output,
        epochs=args.epochs,
        seed=args.seed,
        backend=args.backend,
        compact_checkpoint=args.compact_checkpoint,
    )
    write_json(args.report, result)
    print("Selected using validation only:", result["selected"])
