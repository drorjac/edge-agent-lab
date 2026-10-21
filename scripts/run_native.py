"""Compile the Swift CLI directly and verify its probabilities against Core ML Python."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np

from edge_agent_lab.common import write_json


def main():
    import coremltools as ct

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/apple/sensor_weight_int8.mlpackage")
    )
    parser.add_argument("--input", type=Path, default=Path("artifacts/apple/demo_input.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/native"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sdk = subprocess.run(
        ["xcrun", "--sdk", "macosx", "--show-sdk-path"], check=True, capture_output=True, text=True
    ).stdout.strip()
    source = Path("projects/01-edge-sense/AppleDemo/Sources/ActivityDemo/main.swift")
    binary = args.output / "activity-demo"
    subprocess.run(
        [
            "swiftc",
            "-target",
            "arm64-apple-macosx13.0",
            "-sdk",
            sdk,
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
    )
    completed = subprocess.run(
        [str(binary.resolve()), str(args.model.resolve()), str(args.input.resolve())],
        check=True,
        capture_output=True,
        text=True,
    )
    native = json.loads(completed.stdout)
    sensors = np.array(json.loads(args.input.read_text())["sensors"], dtype="float32")
    model = ct.models.MLModel(str(args.model), compute_units=ct.ComputeUnit.CPU_ONLY)
    logits = model.predict({"sensors": sensors})["logits"][0]
    probabilities = np.exp(logits - logits.max())
    probabilities /= probabilities.sum()
    difference = float(abs(np.array(native["probabilities"]) - probabilities).max())
    parity = {
        "native": native,
        "python_probabilities": probabilities.tolist(),
        "max_probability_difference": difference,
        "passed": difference < 1e-5,
        "compilation": "swiftc direct; arm64 macOS13 SDK",
        "note": "Host SwiftPM PlatformPath is incomplete; direct compilation works.",
    }
    write_json(args.output / "parity.json", parity)
    print(json.dumps(parity, indent=2))
    if not parity["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
