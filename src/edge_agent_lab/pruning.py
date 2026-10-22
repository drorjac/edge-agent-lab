"""Magnitude pruning and physical channel compaction for non-fusion SensorNet models."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn.utils import prune
from torch.utils.data import DataLoader, TensorDataset

from .common import environment, seed_everything, sha256, write_json
from .data import corrupt, ensure_data, load_split
from .inference import resolve_device, run_inference
from .metrics import benchmark, classification, predict
from .models import SensorNet, load_model, save_model


def weight_modules(model):
    return [m for m in model.modules() if isinstance(m, (nn.Conv1d, nn.Linear))]


def apply_magnitude_pruning(model, amount: float) -> None:
    if not 0 < amount < 1:
        raise ValueError("amount must be between zero and one")
    prune.global_unstructured(
        [(m, "weight") for m in weight_modules(model)],
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )


def remove_masks(model) -> None:
    for module in weight_modules(model):
        if hasattr(module, "weight_mask"):
            prune.remove(module, "weight")


def weight_sparsity(model) -> float:
    weights = [m.weight.detach() for m in weight_modules(model)]
    return sum(int((w == 0).sum()) for w in weights) / sum(w.numel() for w in weights)


def _largest_rows(weight, count):
    scores = weight.detach().abs().flatten(1).sum(1)
    return scores.argsort(descending=True, stable=True)[:count].sort().values


def compact_channels(source: SensorNet, width: int) -> tuple[SensorNet, dict]:
    """Slice producer outputs and consumer inputs, including BN and flattened features.

    Only the sequential non-fusion architecture is supported. This removes dense
    dimensions; a structured zero mask alone would leave those dimensions intact.
    """
    if source.fusion or not 0 < width < source.width:
        raise ValueError("Requires a non-fusion model and a smaller positive width")
    if any(hasattr(m, "weight_mask") for m in weight_modules(source)):
        raise ValueError("Compact a floating unmasked model, not a masked pruning run")
    target = SensorNet(width=width, accel_only=source.accel_only).eval()
    first = _largest_rows(source.encoder[0].weight, 2 * width)
    second = _largest_rows(source.encoder[3].weight, 2 * width)
    hidden = _largest_rows(source.head[0].weight, 4 * width)
    flattened = (second[:, None] * 8 + torch.arange(8)[None, :]).flatten()
    with torch.no_grad():
        target.mean.copy_(source.mean)
        target.scale.copy_(source.scale)
        for position, output, inputs in [(0, first, None), (3, second, first)]:
            src, dst = source.encoder[position], target.encoder[position]
            weight = src.weight[output]
            dst.weight.copy_(weight if inputs is None else weight[:, inputs])
            dst.bias.copy_(src.bias[output])
            src_bn, dst_bn = source.encoder[position + 1], target.encoder[position + 1]
            for name in ("weight", "bias", "running_mean", "running_var"):
                getattr(dst_bn, name).copy_(getattr(src_bn, name)[output])
            dst_bn.num_batches_tracked.copy_(src_bn.num_batches_tracked)
        target.head[0].weight.copy_(source.head[0].weight[hidden][:, flattened])
        target.head[0].bias.copy_(source.head[0].bias[hidden])
        target.head[3].weight.copy_(source.head[3].weight[:, hidden])
        target.head[3].bias.copy_(source.head[3].bias)
    return target, {
        "first_channels": first.tolist(),
        "second_channels": second.tolist(),
        "hidden_units": hidden.tolist(),
        "source_width": source.width,
        "target_width": width,
        "ranking": "layerwise output-row L1 magnitude; no test data",
    }


def finetune(model, train, val, epochs: int, seed: int) -> list[dict]:
    """Keep pruning masks active through optimization; choose epochs on validation only."""
    seed_everything(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(train.x), torch.from_numpy(train.y)),
        batch_size=128,
        shuffle=True,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.001)
    best = classification(val.y, predict(model, val.x))["macro_f1"]
    best_state = copy.deepcopy(model.state_dict())
    history = []
    for epoch in range(epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
        metrics = classification(val.y, predict(model, val.x))
        history.append({"epoch": epoch + 1, "validation_macro_f1": metrics["macro_f1"]})
        if metrics["macro_f1"] > best:
            best, best_state = metrics["macro_f1"], copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    # Refresh the pruning pre-hook after restoring weight_orig and masks.
    predict(model, val.x[:1])
    return history


def repeat_timings(candidates: list[dict], x: np.ndarray, seed: int, rounds: int = 3) -> None:
    """Interleave candidates after training to reduce ordering and cold-start bias."""
    rng = np.random.default_rng(seed)
    devices = list(dict.fromkeys(["cpu", str(resolve_device("auto"))]))
    for row in candidates:
        row["timing_rounds"] = []
    for _ in range(rounds):
        for index in rng.permutation(len(candidates)):
            row = candidates[index]
            model = load_model(Path(row["checkpoint"]))
            measurement = {"cpu_batch1": benchmark(model, x), "device_batch64": {}}
            for device in devices:
                _, timing = run_inference(model, x[:256], device, 64, repeats=50, warmup=30)
                measurement["device_batch64"][device] = timing
            row["timing_rounds"].append(measurement)
    for row in candidates:
        samples = row["timing_rounds"]
        row["cpu_batch1"] = dict(samples[0]["cpu_batch1"])
        for metric in ("p50_ms", "p95_ms"):
            row["cpu_batch1"][metric] = float(np.median([s["cpu_batch1"][metric] for s in samples]))
        row["cpu_batch1"]["p95_round_range_ms"] = [
            float(f(s["cpu_batch1"]["p95_ms"] for s in samples)) for f in (min, max)
        ]
        row["cpu_batch1"]["summary"] = "median across interleaved post-training timing rounds"
        for device in devices:
            row["device_batch64"][device] = dict(samples[0]["device_batch64"][device])
            summary = row["device_batch64"][device]
            for metric in ("batch_p50_ms", "batch_p95_ms", "windows_per_second_at_p50"):
                summary[metric] = float(
                    np.median([s["device_batch64"][device][metric] for s in samples])
                )
            summary["summary"] = "median across interleaved post-training timing rounds"


def run_study(checkpoint: Path, data_root: Path, output: Path, epochs=5, seed=7) -> dict:
    if epochs < 0:
        raise ValueError("epochs must be nonnegative")
    seed_everything(seed)
    manifest = ensure_data(data_root)
    train, val = load_split(data_root, "train"), load_split(data_root, "val")
    source = load_model(checkpoint)
    if source.fusion:
        raise ValueError("Use a non-fusion floating student checkpoint")
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "source_checkpoint": str(checkpoint),
        "source_sha256": sha256(checkpoint),
        "dataset_sha256": manifest["archive_sha256"],
        "environment": environment(),
        "seed": seed,
        "fine_tuning_epochs": epochs,
        "selection_split": "validation",
        "test_used_for_selection": False,
        "selection_rule": "smallest state within 0.02 validation macro F1 of original dense source",
        "candidates": [],
        "limits": "Single checkpoint/seed; dense kernels; no sparse-runtime acceleration claim.",
    }
    specs = [
        ("dense_original", None, 0),
        ("dense_finetuned", None, epochs),
        ("magnitude_50", ("magnitude", 0.5), epochs),
        ("magnitude_70", ("magnitude", 0.7), epochs),
        ("channels_75pct", ("channels", max(1, source.width * 3 // 4)), epochs),
        ("channels_50pct", ("channels", max(1, source.width // 2)), epochs),
    ]
    for name, spec, tuning_epochs in specs:
        started = time.monotonic()
        model = load_model(checkpoint)
        details = {}
        if spec and spec[0] == "magnitude":
            apply_magnitude_pruning(model, spec[1])
            details = {"method": "global L1 unstructured", "target_weight_sparsity": spec[1]}
        elif spec:
            model, details = compact_channels(model, spec[1])
        before = classification(val.y, predict(model, val.x))
        history = finetune(model, train, val, tuning_epochs, seed) if tuning_epochs else []
        # Removing reparameterization preserves zeros; dense storage remains dense.
        remove_masks(model)
        path = output / name / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        save_model(model, path, model.config())
        restored = load_model(path)
        np.testing.assert_allclose(predict(model, val.x[:16]), predict(restored, val.x[:16]))
        row = {
            "name": name,
            "details": details,
            "before_finetuning_validation": before,
            "validation": classification(val.y, predict(restored, val.x)),
            "outage_validation": classification(
                val.y, predict(restored, corrupt(val.x, "gyro_dropout", seed=seed))
            ),
            "parameters": sum(p.numel() for p in restored.parameters()),
            "weight_sparsity": weight_sparsity(restored),
            "checkpoint_bytes": path.stat().st_size,
            "checkpoint_sha256": sha256(path),
            "checkpoint": str(path),
            "cpu_batch1": benchmark(restored, val.x),
            "history": history,
            "seconds": time.monotonic() - started,
        }
        # Same input subset, batch size, and synchronization for every device run.
        row["device_batch64"] = {}
        for device in dict.fromkeys(["cpu", str(resolve_device("auto"))]):
            _, timing = run_inference(restored, val.x[:256], device, 64, repeats=30)
            row["device_batch64"][device] = timing
        result["candidates"].append(row)
        write_json(output / "progress.json", result)
        print(name, "validation F1", round(row["validation"]["macro_f1"], 4), flush=True)
    floor = result["candidates"][0]["validation"]["macro_f1"] - 0.02
    feasible = [c for c in result["candidates"] if c["validation"]["macro_f1"] >= floor]
    winner = min(feasible, key=lambda c: (c["checkpoint_bytes"], -c["validation"]["macro_f1"]))
    result["selected"] = winner["name"]
    result["minimum_validation_f1"] = floor
    repeat_timings(result["candidates"], val.x, seed)
    # All pruning ratios, epoch choices, and selection are fixed before loading test.
    test = load_split(data_root, "test")
    for row in result["candidates"]:
        model = load_model(Path(row["checkpoint"]))
        row["test"] = classification(test.y, predict(model, test.x))
    write_json(output / "results.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/edge/seed-7/distilled/model.pt")
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/har"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/pruning"))
    parser.add_argument("--report", type=Path, default=Path("reports/pruning.json"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    result = run_study(args.checkpoint, args.data_root, args.output, args.epochs, args.seed)
    write_json(args.report, result)
    print("Selected using validation only:", result["selected"])
