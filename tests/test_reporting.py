import json

from edge_agent_lab.common import write_json
from edge_agent_lab.reporting import build_reports


def test_portable_report_uses_custom_roots_and_escapes_script_content(tmp_path):
    artifacts = tmp_path / "custom artifacts"
    edge = artifacts / "different-edge-name"
    data = tmp_path / "custom data"
    output = tmp_path / "published report"
    candidate = "</script><script>alert(1)</script>"
    write_json(
        edge / "summary.json",
        {
            "seeds": [7],
            "results": [
                {
                    "name": candidate,
                    "seed": 7,
                    "validation": {"macro_f1": 0.9},
                    "benchmark": {"p95_ms": 0.5, "state_dict_bytes": 1000},
                }
            ],
        },
    )
    write_json(data / "manifest.json", {"dataset": "custom-data"})
    report = build_reports(artifacts, output, data_root=data, edge_root=edge)
    assert report["training_runs"] == 1
    assert json.loads((output / "data_manifest.json").read_text())["dataset"] == "custom-data"
    assert candidate not in (output / "index.html").read_text()
    assert json.loads((output / "results.json").read_text())["aggregates"][0]["name"] == candidate
    assert (output / "model_tradeoffs.png").stat().st_size > 1000
