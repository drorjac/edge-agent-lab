"""Command-line entry points for all three projects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_json, seed_everything


def main():
    parser = argparse.ArgumentParser(
        description="Edge Agent Lab: three reproducible research projects"
    )
    parser.add_argument("--data-root", type=Path, default=Path("data/har"))
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare", help="Fetch, validate, split, and checksum real UCI sensor data")
    train = sub.add_parser("train", help="Run edge model and sim-to-real experiments")
    train.add_argument("--output", type=Path, default=Path("artifacts/edge"))
    train.add_argument("--seeds", type=int, nargs="+", default=[7])
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument(
        "--compact", action="store_true", help="Skip simulator and modality ablations"
    )
    evaluate = sub.add_parser("evaluate", help="Final held-out test and stress evaluation")
    evaluate.add_argument("--layout", choices=["edge", "search"], default="edge")
    evaluate.add_argument("--output", type=Path, default=Path("artifacts/edge"))
    steward = sub.add_parser("steward", help="Run a typed data engineering agent")
    steward.add_argument("task", nargs="?", default="Audit dataset provenance and subject leakage")
    steward.add_argument("--output", type=Path, default=Path("artifacts/steward"))
    steward.add_argument(
        "--planner", choices=["policy", "ollama", "local-intent"], default="policy"
    )
    steward.add_argument(
        "--gguf", type=Path, default=Path("data/models/qwen2.5-1.5b-instruct-q4_k_m.gguf")
    )
    steward.add_argument("--model", default="qwen2.5:3b")
    seval = sub.add_parser("steward-eval", help="Run golden task and retrieval evaluation")
    seval.add_argument("--output", type=Path, default=Path("artifacts/steward"))
    seval.add_argument("--golden", type=Path, default=Path("configs/steward_golden.json"))
    pilot = sub.add_parser(
        "pilot", help="Select an edge candidate under measured deployment constraints"
    )
    pilot.add_argument("--edge-root", type=Path, default=Path("artifacts/edge"))
    pilot.add_argument("--output", type=Path, default=Path("artifacts/pilot"))
    pilot.add_argument("--objective", type=Path, default=Path("configs/pilot.json"))
    pilot.add_argument("--seed", type=int, default=7)
    search = sub.add_parser(
        "search", help="Train new candidates under a bounded resource objective"
    )
    search.add_argument("--objective", type=Path, default=Path("configs/search.json"))
    search.add_argument(
        "--teacher", type=Path, default=Path("artifacts/edge/seed-7/teacher/model.pt")
    )
    search.add_argument("--output", type=Path, default=Path("artifacts/search"))
    search.add_argument("--epochs", type=int, choices=range(1, 81), metavar="EPOCHS", default=30)
    search.add_argument("--seed", type=int, default=7)
    probe = sub.add_parser("probe", help="Independent real-data HAR70+ transfer evaluation")
    probe.add_argument("--seed", type=int, default=7)
    probe.add_argument("--root", type=Path, default=Path("data/har70"))
    probe.add_argument("--edge-root", type=Path, default=Path("artifacts/edge"))
    probe.add_argument("--output", type=Path, default=Path("artifacts/external"))
    baseline = sub.add_parser(
        "baselines", help="Train statistical logistic and boosting references"
    )
    baseline.add_argument("--output", type=Path, default=Path("artifacts/baselines"))
    apple = sub.add_parser("export-apple", help="Export and benchmark Core ML artifacts on macOS")
    apple.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/edge/seed-7/distilled/model.pt")
    )
    apple.add_argument("--output", type=Path, default=Path("artifacts/apple"))
    apple.add_argument("--limit", type=int, default=0, help="0 evaluates all validation rows")
    report = sub.add_parser(
        "report", help="Generate portable Markdown, figures, and static HTML dashboard"
    )
    report.add_argument("--edge-root", type=Path, default=None)
    report.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    report.add_argument("--output", type=Path, default=Path("reports"))
    args = parser.parse_args()
    seed_everything(7)
    if args.command == "prepare":
        from .data import prepare

        result = prepare(args.data_root)
    elif args.command == "train":
        from .training import run_edge

        if args.epochs < 1:
            parser.error("--epochs must be positive")
        result = run_edge(args.data_root, args.output, args.seeds, args.epochs, not args.compact)
    elif args.command == "evaluate":
        from .training import evaluate_test

        result = evaluate_test(args.data_root, args.output, args.layout)
    elif args.command == "steward":
        from .steward import Steward

        agent = Steward(args.data_root, args.output)
        if args.planner == "local-intent":
            if not args.gguf.exists():
                parser.error("GGUF missing; run scripts/evaluate_intent_router.py to fetch it")
            from .local_llm import IntentRouter

            result = agent.run(
                args.task,
                planner="local_llm_intent_router",
                action_provider=IntentRouter(args.gguf),
            )
        else:
            result = agent.run(args.task, planner=args.planner, ollama_model=args.model)
    elif args.command == "steward-eval":
        from .steward import evaluate_steward

        result = evaluate_steward(args.data_root, args.output, args.golden)
    elif args.command == "pilot":
        from .pilot import Objective, run_pilot, stream_monitor

        result = run_pilot(
            args.data_root,
            args.edge_root,
            args.output,
            Objective.model_validate(read_json(args.objective)),
            args.seed,
        )
        stream_monitor(args.data_root, args.output / "selection.json", args.output)
    elif args.command == "search":
        from .pilot import Objective
        from .search import run_search

        result = run_search(
            args.data_root,
            args.teacher,
            args.output,
            Objective.model_validate(read_json(args.objective)),
            args.epochs,
            args.seed,
        )
    elif args.command == "probe":
        from .external import evaluate_external

        result = evaluate_external(args.root, args.edge_root, args.output, args.seed)
    elif args.command == "baselines":
        from .baselines import run_baselines

        result = run_baselines(args.data_root, args.output)
    elif args.command == "export-apple":
        from .apple import export_apple

        result = export_apple(args.checkpoint, args.data_root, args.output, args.limit)
    else:
        from .reporting import build_reports

        result = build_reports(args.artifacts, args.output, args.data_root, args.edge_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
