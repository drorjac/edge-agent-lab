"""Evaluate a refined tool contract without fixing the model's incident decisions."""

from pathlib import Path

from edge_agent_lab.local_llm import FILENAME, evaluate_local

evaluate_local(
    Path("data/har"),
    Path("artifacts/steward-grounded"),
    Path("data/models") / FILENAME,
    Path("configs/steward_golden.json"),
    grounded_tools=True,
)
