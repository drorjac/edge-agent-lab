import argparse
from pathlib import Path

import yaml

from edge_agent_lab.common import seed_everything
from edge_agent_lab.training import run_edge

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/overnight.yaml")
args = parser.parse_args()
config = yaml.safe_load(Path(args.config).read_text())
seed_everything(config["seeds"][0])
run_edge(
    Path(config["data_root"]),
    Path(config["output"]),
    config["seeds"],
    config["epochs"],
    config["extended"],
)
