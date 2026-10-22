"""Floating-point sensor inference on CPU, CUDA, or Apple MPS."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from edge_agent_lab.common import sha256
from edge_agent_lab.data import ensure_data, load_split
from edge_agent_lab.metrics import classification
from edge_agent_lab.models import load_model


def resolve_device(requested: str = "auto") -> torch.device:
    if requested not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError("device must be auto, cpu, cuda, or mps")
    cuda = torch.cuda.is_available()
    mps = torch.backends.mps.is_available()
    if requested == "auto":
        requested = "cuda" if cuda else "mps" if mps else "cpu"
    if requested == "cuda" and not cuda:
        raise RuntimeError(
            "CUDA is unavailable. Install a compatible CUDA PyTorch build or use cpu."
        )
    if requested == "mps" and not mps:
        raise RuntimeError("Apple MPS is unavailable on this machine. Use cpu instead.")
    return torch.device(requested)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


@torch.inference_mode()
def run_inference(
    model,
    x: np.ndarray,
    device: str = "auto",
    batch_size: int = 64,
    repeats: int = 50,
    warmup: int = 10,
) -> tuple[np.ndarray, dict]:
    """Return probabilities and synchronized, resident-input forward timings.

    Copies the caller's floating model. Prediction transfers are outside the timed
    region. Timing measures a full batch, including dispatch and synchronization,
    rather than claiming per-request latency from batched throughput.
    """
    if batch_size < 1 or repeats < 1 or warmup < 0:
        raise ValueError("batch_size and repeats must be positive; warmup must be nonnegative")
    if x.ndim != 3 or x.shape[1:] != (9, 128) or len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("Expected nonempty finite sensor windows with shape (N, 9, 128)")
    if any("quantized" in type(module).__module__ for module in model.modules()):
        raise ValueError("Use a floating checkpoint for this benchmark; dynamic INT8 is CPU-only.")
    target = resolve_device(device)
    net = copy.deepcopy(model).float().eval().to(target)
    probabilities = []
    for start in range(0, len(x), batch_size):
        inputs = torch.as_tensor(x[start : start + batch_size], dtype=torch.float32, device=target)
        probabilities.append(net(inputs).softmax(1).cpu())
    sample = torch.as_tensor(x[:batch_size], dtype=torch.float32, device=target)
    for _ in range(warmup):
        net(sample)
    synchronize(target)
    durations = []
    for _ in range(repeats):
        synchronize(target)
        start = time.perf_counter_ns()
        net(sample)
        synchronize(target)
        durations.append((time.perf_counter_ns() - start) / 1e6)
    name = (
        torch.cuda.get_device_name(target)
        if target.type == "cuda"
        else "Apple GPU (MPS)"
        if target.type == "mps"
        else "CPU"
    )
    p50 = float(np.median(durations))
    return torch.cat(probabilities).numpy(), {
        "device": str(target),
        "hardware": name,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cpu_threads": torch.get_num_threads(),
        "dtype": "float32",
        "rows": len(x),
        "batch_size": len(sample),
        "repeats": repeats,
        "warmup": warmup,
        "batch_p50_ms": p50,
        "batch_p95_ms": float(np.percentile(durations, 95)),
        "windows_per_second_at_p50": len(sample) * 1000 / p50,
        "timing_scope": (
            "resident-input forward; includes dispatch and synchronization; no transfers"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--data-root", type=Path, default=Path("data/har"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", type=Path, default=Path("artifacts/inference/latest.json"))
    args = parser.parse_args()
    # Fail clearly before touching data if the requested hardware is unavailable.
    resolve_device(args.device)
    if "dynamic_int8" in args.checkpoint.parts:
        parser.error("Choose a floating student, distilled, teacher, or trained-search checkpoint.")
    ensure_data(args.data_root)
    split = load_split(args.data_root, args.split)
    model = load_model(args.checkpoint)
    probabilities, timing = run_inference(
        model, split.x, args.device, args.batch_size, args.repeats
    )
    cpu_probabilities, cpu_timing = run_inference(
        model, split.x, "cpu", args.batch_size, args.repeats
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha256(args.checkpoint),
        "split": args.split,
        "split_sha256": sha256(args.data_root / f"{args.split}.npz"),
        "classification": classification(split.y, probabilities),
        "timing": timing,
        "cpu_reference_timing": cpu_timing,
        "cpu_top1_agreement": float(
            (probabilities.argmax(1) == cpu_probabilities.argmax(1)).mean()
        ),
        "max_probability_difference": float(np.abs(probabilities - cpu_probabilities).max()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
