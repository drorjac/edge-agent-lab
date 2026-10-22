# 03 · Edge Pilot

[Executed summary notebook](../../notebooks/03_edge_pilot.ipynb)

**An engineering workflow that selects a compact model under measured constraints.**

The controller combines data quality, experiment history, tool-based measurements,
and explicit promotion gates. A planner proposes the next candidate, a runner
measures it, and a critic reports constraint margins. Following a failed gate,
the policy prioritizes robust candidates. The output is a feasible candidate or
an explicit refusal to select one, plus a Pareto frontier and trace.

## Run

```bash
edge-lab prepare
edge-lab train --seeds 7 --epochs 40 --compact
edge-lab pilot --objective configs/pilot.json
edge-lab report
```

The default objective caps warmed PyTorch CPU p95 latency and serialized state
size, requires validation macro F1, and applies an additional missing-gyroscope
robustness gate. Edit the JSON objective to explore another resource envelope.
The resource values are illustrative development constraints, not a hardware SLA.

## Workflow

1. Run Data Steward preflight against the checksum-verified dataset.
2. Rank trained candidate metadata using validation quality and storage priors.
3. Load each proposed checkpoint and measure clean/dropout validation F1 and latency.
4. Apply every hard gate; record acceptance and numerical margins.
5. Continue within trial/time budgets; prioritize robust training after failures.
6. Select from feasible candidates using 70% clean and 30% dropout validation F1.
7. Replay labeled validation batches with a synthetic gyroscope outage halfway through.
8. Log alerts and recommended inspection actions; generate portable evidence.

The workflow actors are explicit software components. This is a bounded adaptive
policy over trained experiments, not a claim of autonomous scientific discovery
or independent LLM agents. Actual local LLM planning is measured separately in
Data Steward.

## Evaluation boundaries

Test results never influence ranking. Unit tests enforce each hard gate, verify
that changing test metrics cannot affect selection, preserve nondominated
tradeoffs, and reject invalid objectives. The replay is labeled validation data
with a synthetic incident, not live production telemetry. An alert prompts
inspection; it does not trigger unsupervised retraining or real deployment.

Inspect [`pilot.py`](../../src/edge_agent_lab/pilot.py), the
[objective](../../configs/pilot.json), and [measured results](../../reports/RESULTS.md).

## On-demand experiment path

```bash
edge-lab search --objective configs/search.json
```

The tighter 32,000-byte state objective is below the existing distilled/INT8
candidate sizes. The search estimates storage from untrained architecture states,
proposes the largest fitting width, and **actually trains** distilled candidates.
It compares robust and unaugmented treatments and can prioritize another robust
proposal after a dropout-gate failure. All proposals have allowlisted widths,
bounded epochs, a seed, and a trial/time budget. Accuracy is measured after
training, never inferred from the storage estimate.

Saved outputs include training histories, new checkpoints, proposal records,
critic margins, selection, and the teacher/dataset hashes. Existing trained trials
can be reused only under an identical search specification. Time is checked
between training jobs; an in-flight job can exceed the remaining time budget.
