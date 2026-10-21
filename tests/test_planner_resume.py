import pytest

from edge_agent_lab.common import sha256, write_json
from edge_agent_lab.local_llm import evaluate_local


def completed_fixture(tmp_path):
    model = tmp_path / "model.gguf"
    model.write_bytes(b"the cache guard must run before a model provider is imported")
    golden = tmp_path / "golden.json"
    write_json(golden, [{"id": "done"}])
    output = tmp_path / "evaluation"
    report = {
        "model_sha256": sha256(model),
        "golden_sha256": sha256(golden),
        "guarded": True,
        "policy_guard": False,
        "threads": 2,
        "gpu_layers": 0,
        "results": [{"id": "done"}],
        "elapsed_seconds": 1,
        "cases": 1,
    }
    write_json(output / "local_llm_evaluation.json", report)
    return model, golden, output, report


def test_completed_planner_evaluation_resumes_without_loading_optional_model(tmp_path):
    model, golden, output, report = completed_fixture(tmp_path)
    assert evaluate_local(tmp_path / "unused-data", output, model, golden) == report


def test_planner_resume_rejects_changed_suite_or_policy(tmp_path):
    model, golden, output, _ = completed_fixture(tmp_path)
    with pytest.raises(ValueError, match="configuration changed"):
        evaluate_local(tmp_path / "unused-data", output, model, golden, policy_guard=True)
    write_json(golden, [{"id": "new-task"}])
    with pytest.raises(ValueError, match="configuration changed"):
        evaluate_local(tmp_path / "unused-data", output, model, golden)


def test_grounded_tool_contract_binds_query_and_scopes_arguments():
    from edge_agent_lab.local_llm import grounded_tool_schema

    task = 'Inspect "quoted" source and license'
    choices = grounded_tool_schema(["retrieve", "finish"], task)["anyOf"]
    retrieve, finish = choices
    assert retrieve["properties"]["query"]["enum"] == [task]
    assert "query" in retrieve["required"]
    assert "split" not in retrieve["properties"] and "corruption" not in retrieve["properties"]
    assert set(finish["properties"]) == {"tool", "reason"}
    assert not finish["additionalProperties"]


def test_incident_tool_contract_keeps_probes_on_validation():
    from edge_agent_lab.local_llm import grounded_tool_schema

    spec = grounded_tool_schema(["simulate_shift"], "gyro outage")["anyOf"][0]
    assert spec["properties"]["split"]["enum"] == ["val"]
    assert "gyro_dropout" in spec["properties"]["corruption"]["enum"]
    assert set(spec["required"]) == {"tool", "reason", "split", "corruption"}
