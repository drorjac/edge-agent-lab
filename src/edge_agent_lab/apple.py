"""Core ML export, weight compression, parity, and measured local runtime."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from .common import environment, sha256, write_json
from .data import load_split
from .metrics import classification, predict
from .models import load_model


def export_apple(checkpoint: Path, data_root: Path, output: Path, limit: int = 0) -> dict:
    import coremltools as ct
    import coremltools.optimize.coreml as cto

    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    model = load_model(checkpoint)
    val = load_split(data_root, "val")
    if limit > 0:
        indices = np.random.default_rng(7).permutation(len(val.y))[:limit]
        val.x, val.y, val.subjects = val.x[indices], val.y[indices], val.subjects[indices]
    limit = len(val.y)
    write_json(output / "demo_input.json", {"sensors": val.x[:1].tolist(), "label": int(val.y[0])})
    example = torch.from_numpy(val.x[:1])
    traced = torch.jit.trace(model, example)
    converted = ct.convert(
        traced,
        inputs=[ct.TensorType(name="sensors", shape=(1, 9, 128))],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT32,
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    converted.short_description = (
        "Six-class activity recognition with embedded train-only normalization"
    )
    converted.author = "Edge Agent Lab"
    converted.user_defined_metadata["input"] = (
        "float32 [1,9,128], body_acc xyz, body_gyro xyz, total_acc xyz"
    )
    converted.save(str(output / "sensor_float32.mlpackage"))
    config = cto.OptimizationConfig(
        global_config=cto.OpLinearQuantizerConfig(
            mode="linear_symmetric", dtype=np.int8, weight_threshold=64
        )
    )
    compressed = cto.linear_quantize_weights(converted, config)
    compressed.save(str(output / "sensor_weight_int8.mlpackage"))
    reference = predict(model, val.x[:limit])
    variants = []
    for name, path in [
        ("float32", output / "sensor_float32.mlpackage"),
        ("weight_int8", output / "sensor_weight_int8.mlpackage"),
    ]:
        size = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
        row = {"variant": name, "package_bytes": size, "runtime": "not_measured"}
        try:
            # Explicit CPU_ONLY establishes a reproducible backend; ALL is a separate measurement.
            runtime = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.CPU_ONLY)
            for _ in range(10):
                runtime.predict({"sensors": val.x[:1]})
            logits, timings = [], []
            for x in val.x[:limit]:
                start = time.perf_counter_ns()
                value = runtime.predict({"sensors": x[None]})["logits"]
                timings.append((time.perf_counter_ns() - start) / 1e6)
                logits.append(value[0])
            array = torch.tensor(np.array(logits)).softmax(1).numpy()
            row.update(
                {
                    "runtime": "measured",
                    "compute_units": "CPU_ONLY",
                    "validation_rows": len(array),
                    "classification": classification(val.y[:limit], array),
                    "p50_ms": float(np.median(timings)),
                    "p95_ms": float(np.percentile(timings, 95)),
                    "pytorch_top1_agreement": float(
                        (array.argmax(1) == reference.argmax(1)).mean()
                    ),
                    "max_probability_difference": float(abs(array - reference).max()),
                }
            )
        except Exception as exc:
            row["runtime_error"] = f"{type(exc).__name__}: {exc}"
        variants.append(row)
    report = {
        "source_checkpoint": str(checkpoint),
        "source_sha256": sha256(checkpoint),
        "coremltools_version": ct.__version__,
        "input_shape": [1, 9, 128],
        "minimum_target": "macOS13",
        "variants": variants,
        "environment": environment(),
        "quantization_scope": "Core ML weight-only int8; activations remain float",
        "limitations": "CPU_ONLY timing includes Python overhead; ANE placement is unmeasured.",
    }
    write_json(output / "export.json", report)
    return report
