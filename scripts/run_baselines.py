from pathlib import Path

from edge_agent_lab.baselines import run_baselines

run_baselines(Path("data/har"), Path("artifacts/baselines"))
