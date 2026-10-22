# 01 · Edge Sense

[Executed summary notebook](../../notebooks/01_edge_sense.ipynb)

[Pruning experiment](../../docs/PRUNING.md): global magnitude pruning and physical channel compaction, with masked fine-tuning, export checks, and measured size/quality/timing comparisons. Run `edge-prune --epochs 5` after generating the floating distilled checkpoint.

**Efficient activity recognition from accelerometer and gyroscope windows.**

Build a real-data pipeline, train a multistream teacher and compact baselines,
compress the student, measure domain-shift robustness, and export to Core ML.
The study includes a procedural motion simulator and an equal-label-budget
simulation-to-real comparison.

## Run

From the repository root after installation:

```bash
edge-lab prepare
edge-lab train --seeds 7 23 42 --epochs 40
edge-lab evaluate
edge-lab report
```

Training selection uses held-out validation subjects. `evaluate` reads the official
test partition for the final report; its metrics never feed candidate ranking.
For a smaller run use `--compact --seeds 7`, or a separate output directory with
fewer epochs. Full runs reuse completed candidates in their output directory.

## What to inspect

- [`models.py`](../../src/edge_agent_lab/models.py): multistream teacher, compact student, embedded normalization, restricted quantized checkpoint format.
- [`training.py`](../../src/edge_agent_lab/training.py): supervised learning, distillation, augmentation, equal-label adaptation, final test evaluation.
- [`data.py`](../../src/edge_agent_lab/data.py): source hash, subject splits, procedural priors, physically interpretable corruptions.
- [Protocol](../../docs/EXPERIMENTS.md) and [results](../../reports/RESULTS.md): actual comparisons and limitations.

## Apple deployment

```bash
uv pip install -e '.[apple]'
edge-lab export-apple
# Model packages + validation inference input are written to artifacts/apple/
swift run --package-path projects/01-edge-sense/AppleDemo activity-demo \
  "$PWD/artifacts/apple/sensor_weight_int8.mlpackage" \
  "$PWD/artifacts/apple/demo_input.json"
```

The native Swift example uses Core ML, checks the input shape and finiteness,
and reports class probabilities. It requires a working Apple SDK installation.
Python Core ML export and runtime evaluation are independent of Swift compilation.
Normalization is in the model; supply original sensor units in the documented
channel order. Weight-only compression leaves activations floating point.

The model classifies six activities from a phone near the waist. It is not a
medical device, a general movement detector, or validated for arbitrary wearables.

## Optional CUDA / Apple GPU inference

After generating the floating checkpoint and preparing the data:

```bash
edge-infer --checkpoint artifacts/edge/seed-7/distilled/model.pt --device auto --batch-size 64
```

`auto` selects CUDA, then Apple MPS, then CPU. Use `--device cuda` to require
NVIDIA execution, `--device mps` for an Apple GPU, or `--device cpu` for CPU.
CUDA requires a compatible CUDA-enabled PyTorch build and GPU driver. The
PyTorch 2.2.2 study pin may not support recent GPU architectures; use a separately
validated environment when needed. Dynamic INT8 checkpoints remain CPU-only.

The command measures floating-point forward passes with inputs already on the
device and synchronization around each call. It reports batch p50/p95,
throughput, validation classification metrics, and agreement against a CPU
reference. Choose batch size 1 for single-window calls. Output is saved to
`artifacts/inference/latest.json`; existing study reports are unchanged. The
[notebook](../../notebooks/01_edge_sense.ipynb) includes a smaller interactive run.
