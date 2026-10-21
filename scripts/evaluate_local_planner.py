"""Fetch a pinned public GGUF and evaluate genuine local LLM tool planning."""

import argparse
from pathlib import Path

from edge_agent_lab.local_llm import FILENAME, evaluate_local, fetch_model

parser = argparse.ArgumentParser()
parser.add_argument("--data-root", type=Path, default=Path("data/har"))
parser.add_argument("--model-root", type=Path, default=Path("data/models"))
parser.add_argument("--output", type=Path, default=Path("artifacts/steward"))
parser.add_argument("--golden", type=Path, default=Path("configs/steward_golden.json"))
args = parser.parse_args()
fetch_model(args.model_root)
evaluate_local(args.data_root, args.output, args.model_root / FILENAME, args.golden)
