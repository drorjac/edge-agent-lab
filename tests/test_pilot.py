from copy import deepcopy

import pytest
from pydantic import ValidationError

from edge_agent_lab.pilot import Objective, feasible, pareto_front, rank_candidates


def candidate(name="a", f1=0.9, latency=1, size=100_000, dropout=0.7):
    return {
        "candidate": name,
        "validation": {"macro_f1": f1},
        "benchmark": {"p95_ms": latency, "state_dict_bytes": size},
        "dropout_macro_f1": dropout,
    }


@pytest.mark.parametrize(
    "change", [{"f1": 0.5}, {"latency": 9}, {"size": 999_999}, {"dropout": 0.1}]
)
def test_each_constraint_is_a_hard_gate(change):
    assert not feasible(candidate(**change), Objective())


def test_no_feasible_candidate_is_not_promoted():
    assert rank_candidates([candidate(f1=0.1), candidate(size=9_000_000)], Objective()) == []


def test_test_metrics_cannot_affect_ranking():
    a, b = candidate("a", f1=0.94), candidate("b", f1=0.90)
    original = rank_candidates([a, b], Objective())[0]["candidate"]
    changed = deepcopy([a, b])
    changed[0]["test"] = {"accuracy": 0}
    changed[1]["test"] = {"accuracy": 1}
    assert rank_candidates(changed, Objective())[0]["candidate"] == original


def test_pareto_preserves_tradeoff_and_removes_dominated():
    rows = [
        candidate("accurate", f1=0.96, latency=1.5),
        candidate("fast", f1=0.9, latency=0.4),
        candidate("bad", f1=0.8, latency=2),
    ]
    assert set(pareto_front(rows)) == {"accurate", "fast"}


def test_objective_rejects_nonfinite_and_negative_budgets():
    with pytest.raises(ValidationError):
        Objective(max_p95_ms=float("nan"))
    with pytest.raises(ValidationError):
        Objective(max_trials=0)
