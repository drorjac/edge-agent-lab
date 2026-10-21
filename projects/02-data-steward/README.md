# 02 · Data Steward

**A data engineering agent whose decisions can be inspected and evaluated.**

The agent resolves a dataset version, retrieves engineering evidence, audits
schema/completeness/subject isolation, reproduces sensor incidents, and records
every action in SQLite. A deterministic baseline runs without external services;
an optional local Qwen planner exercises the identical typed tools.

## Run

```bash
edge-lab prepare
edge-lab steward "Investigate a missing gyroscope sensor outage"
edge-lab steward-eval
uv pip install -e '.[local-llm]'
python scripts/evaluate_intent_router.py
```

The optional model fetch resolves a publisher commit, checks its published LFS
SHA256, and caches approximately 1.1 GB outside Git. The inference engine can
require a C++ build toolchain. An optional `--planner ollama --model MODEL` adapter
is also available when a local Ollama server is already configured.

## Contracts and behavior

| Tool | Inputs | Output |
|---|---|---|
| catalog | none | Dataset source, license, checksums, subjects, class counts |
| retrieve | bounded text query | Ranked evidence chunks with stable IDs |
| audit | train/val/test name | Schema, nonfinite values, constant channels, split isolation |
| simulate_shift | split and allowlisted corruption | Synthetic incident audit compared with reference statistics |
| finish | reason | Workflow completion signal, accepted only after required evidence steps |

Retrieval fuses TF-IDF cosine and exact-token rankings with reciprocal-rank fusion.
It is a small, transparent runbook corpus, not a dense vector database or a claim
of enterprise search scale. The catalog tracks dataset versions, runs, and tool
observations. SQL values use parameters; the model has no SQL or shell tool.

## Evaluation

Golden tasks cover provenance, leakage, missing sensors, drift, simulator
interpretation, deployment gates, instruction injection, and robustness. Pass
requires relevant retrieval, required tool execution, and a completed workflow.
Invalid JSON, extra fields, missing evidence, repetition beyond the step budget,
and tool errors cannot count as success.

The local planner has explicit workflow guards: required evidence tools cannot be
skipped, completed tools cannot be repeated, and all arguments must be supplied.
These guards are part of the system, not model reasoning capability. The initial
schema-only trial is retained separately when available; it demonstrates why JSON
validity alone does not establish task completion.

See [results](../../reports/RESULTS.md), [source](../../src/edge_agent_lab/steward.py),
and [local planner](../../src/edge_agent_lab/local_llm.py).

## Policy-fixed LLM ablation

```bash
python scripts/evaluate_policy_guarded.py
```

This deliberately removes model authority over tools and arguments. The model
realizes the known plan and supplies a short reason through a constrained grammar.
Compare it with schema-only and workflow-only planning: correctness from fixed
policy must not be presented as autonomous reasoning. The measured cost difference
helps decide whether an LLM adds value to a bounded engineering task.

Portable actual traces are committed under [`reports/traces`](../../reports/traces/).
The optional larger-model comparison uses `scripts/fetch_large_planner.py` and
`scripts/evaluate_large_planner.py`; it downloads about 4.7 GB. Run large-model
inference separately from training on memory-constrained machines.

## Semantic intent route

```bash
python scripts/evaluate_intent_router.py
edge-lab steward --planner local-intent "Investigate a missing gyroscope sensor"
```

The model classifies incident intent **before** retrieval. Software then binds the
original query, runs the typed catalog/retrieval/audit workflow, and executes only
the incident chosen by the model. Retrieved runbook instructions cannot change the
already inferred intent. This bounded decomposition is the recommended local-model
path; it is not long-horizon model planning.

Reproduce loop-style ablations with `scripts/evaluate_local_planner.py` and
`scripts/evaluate_grounded_planner.py`. Per-tool argument scoping fixes retrieval
binding and split selection but still leaves semantic decisions to the model.
All regression scores remain in the report, including unsuccessful variants.
