from pathlib import Path

from edge_agent_lab.common import read_json
from edge_agent_lab.pilot import Objective
from edge_agent_lab.search import run_search

run_search(
    Path("data/har"),
    Path("artifacts/edge/seed-7/teacher/model.pt"),
    Path("artifacts/search"),
    Objective.model_validate(read_json(Path("configs/search.json"))),
)
