# Reproducing the study

## Environment

Use Python 3.11 or 3.12. The development host is Apple silicon, 16 GB RAM, macOS
13.4. The package pins a compatible PyTorch/NumPy baseline. Core ML conversion
requires the `apple` extra, including a setuptools version supplying the
`distutils` compatibility import needed by this Core ML Tools release.

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e '.[dev,apple]'
```

On Linux, install the CPU-specific PyTorch wheel first if CUDA is unnecessary:

```bash
python -m pip install torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
```

## Full workflow

```bash
python scripts/run_all.py --config configs/overnight.yaml --apple --external
# Optional local LLM inference requires its extra; approximately 1.1 GB model.
uv pip install -e '.[local-llm]'
python scripts/evaluate_intent_router.py
edge-lab report
```

The orchestrator records each stage, exit status, elapsed time, command, and log
under `artifacts/orchestration/`. Training reuses complete candidates; failed
candidates restart. Rerunning with a different epoch budget requires a separate
output directory to avoid mixing incompatible studies. An interrupted epoch does
not have a resumable optimizer checkpoint.

The full default matrix has 27 trained floating models and three derived INT8
models. Exact duration depends on CPU load. Four PyTorch threads bound default
parallelism. The experiments are small enough for CPU execution but are not CI
jobs. CI uses synthetic inputs and never downloads real datasets or model weights.

## Native deployment verification

```bash
edge-lab export-apple
python scripts/run_native.py
```

The native verification script compiles the Swift CLI directly and checks its
probabilities against Python Core ML. On the development host, SwiftPM cannot
resolve `PlatformPath` from the installed command-line tools; direct `swiftc`
compilation works. A fully configured Xcode installation can also use the included
Swift package.

## Isolated timing

Stop training and local LLM inference before collecting final deployment timing.
Rerun `edge-lab pilot` to remeasure every explored candidate, then
`edge-lab export-apple` for Core ML. Record machine, power mode, requested backend,
thread count, warmup count, batch size, p50, and p95. These numbers describe local
inference API latency and should not be presented as mobile battery or ANE results.

## Auditing results

- `data/har/manifest.json`: source and derived-array hashes, subject lists, class counts.
- `artifacts/edge/seed-*/NAME/history.json`: epoch loss and validation metrics.
- `artifacts/edge/seed-*/NAME/result.json`: checkpoint hash, validation, timing, environment.
- `artifacts/edge/test_evaluation.json`: reporting-only test metrics and subject bootstrap.
- `artifacts/steward/catalog.sqlite`: dataset versions, runs, actions, observations.
- `artifacts/steward/runs/*.json`: portable agent traces including failures.
- `artifacts/pilot/selection.json`: objective, critic margins, feasible winner or none.
- `artifacts/apple/export.json`: conversion, package sizes, parity, actual runtime status.
- `reports/results.json`: portable snapshot of measured evidence committed to Git.

Raw data, checkpoints, GGUF weights, SQLite files, logs, and build products are
ignored by Git. Network access is needed for first fetch and dependency install;
after caches exist, training, inference, audits, selection, and reporting run
locally.

## Locked dependencies and report verification

`uv.lock` records exact resolved dependency versions. Use
`uv sync --locked --extra dev --extra apple --extra local-llm` for the full macOS
environment, or omit platform/LLM extras as appropriate. Optional native builds
may require a compiler. `uv lock --check` verifies that the lock matches metadata.

After the full report exists, `python scripts/verify_report.py` validates local
image assets and exercises sorting, tabs, escaping, and trace display in a JavaScript
engine using a minimal fake DOM. It uses Node.js when available or macOS
JavaScriptCore. This checks logic and syntax, not browser rendering. No browser
was available through the computer-use integration during the development run;
figures were visually inspected separately.
