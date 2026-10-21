"""Resumable stage orchestration with explicit failures and no unattended prompts."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import yaml

from edge_agent_lab.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/overnight.yaml")
    parser.add_argument("--local-llm", action="store_true")
    parser.add_argument("--apple", action="store_true")
    parser.add_argument("--external", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=Path("reports"))
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    edge = Path(config["output"])
    data = Path(config["data_root"])
    base = edge.parent
    seed = config["seeds"][0]
    root = base / "orchestration"
    root.mkdir(parents=True, exist_ok=True)
    ledger_path = root / "stages.json"
    outcomes = read_json(ledger_path) if ledger_path.exists() else []
    cli = [sys.executable, "-m", "edge_agent_lab.cli", "--data-root", str(data)]
    teacher = edge / f"seed-{seed}/teacher/model.pt"
    distilled = edge / f"seed-{seed}/distilled/model.pt"
    stages = [
        ("training", [sys.executable, "scripts/train_study.py", "--config", args.config], 14400),
        ("baselines", cli + ["baselines", "--output", str(base / "baselines")], 1200),
        ("steward", cli + ["steward-eval", "--output", str(base / "steward")], 600),
        (
            "pilot",
            cli
            + [
                "pilot",
                "--edge-root",
                str(edge),
                "--seed",
                str(seed),
                "--output",
                str(base / "pilot"),
            ],
            1200,
        ),
        (
            "search",
            cli
            + [
                "search",
                "--teacher",
                str(teacher),
                "--seed",
                str(seed),
                "--output",
                str(base / "search"),
            ],
            3600,
        ),
        ("test", cli + ["evaluate", "--output", str(edge)], 1800),
    ]
    stages.append(
        (
            "search_test",
            cli + ["evaluate", "--layout", "search", "--output", str(base / "search")],
            1800,
        )
    )
    if args.external:
        stages.append(
            (
                "external",
                cli
                + [
                    "probe",
                    "--root",
                    str(data.parent / "har70"),
                    "--edge-root",
                    str(edge),
                    "--seed",
                    str(seed),
                    "--output",
                    str(base / "external"),
                ],
                1800,
            )
        )
    if args.local_llm:
        stages.append(
            (
                "local_llm",
                [
                    sys.executable,
                    "scripts/evaluate_intent_router.py",
                    "--data-root",
                    str(data),
                    "--output",
                    str(base / "steward-intent"),
                ],
                3600,
            )
        )
    if args.apple and platform.system() == "Darwin":
        stages.append(
            (
                "coreml",
                cli
                + ["export-apple", "--checkpoint", str(distilled), "--output", str(base / "apple")],
                1800,
            )
        )
    stages += [
        (
            "reports",
            cli
            + [
                "report",
                "--artifacts",
                str(base),
                "--edge-root",
                str(edge),
                "--output",
                str(args.report_dir),
            ],
            600,
        ),
        ("tests", [sys.executable, "-m", "pytest", "-q"], 300),
        ("lint", [sys.executable, "-m", "ruff", "check", "src", "tests", "scripts"], 120),
    ]
    failed = set()
    for name, command, timeout in stages:
        dependencies = (
            {"training"}
            if name in {"pilot", "search", "test", "search_test", "coreml", "reports", "external"}
            else set()
        )
        if dependencies & failed:
            outcomes.append(
                {"stage": name, "status": "blocked", "dependencies": sorted(dependencies & failed)}
            )
            failed.add(name)
            write_json(ledger_path, outcomes)
            continue
        print(f"Starting {name}", flush=True)
        started = time.monotonic()
        try:
            with (root / f"{name}.log").open("w") as log:
                process = subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False
                )
            row = {
                "stage": name,
                "status": "passed" if process.returncode == 0 else "failed",
                "returncode": process.returncode,
                "seconds": time.monotonic() - started,
                "command": command,
            }
        except subprocess.TimeoutExpired:
            row = {"stage": name, "status": "timeout", "seconds": time.monotonic() - started}
        outcomes.append(row)
        if row["status"] != "passed":
            failed.add(name)
        write_json(ledger_path, outcomes)
        print(f"{name}: {row['status']}", flush=True)
    if failed:
        print("Stages requiring attention: " + ", ".join(sorted(failed)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
