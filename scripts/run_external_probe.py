from pathlib import Path

from edge_agent_lab.common import seed_everything
from edge_agent_lab.external import evaluate_external

seed_everything(7)
evaluate_external(Path("data/har70"), Path("artifacts/edge"), Path("artifacts/external"))
