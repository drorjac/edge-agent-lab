"""Measure policy-fixed LLM realization, explicitly not autonomous model planning."""

from pathlib import Path

from edge_agent_lab.local_llm import FILENAME, evaluate_local

evaluate_local(
    Path("data/har"),
    Path("artifacts/steward-policy-guarded"),
    Path("data/models") / FILENAME,
    Path("configs/steward_golden.json"),
    policy_guard=True,
)
