# Pruning the activity-recognition network

The project already uses teacher–student distillation, partial dynamic INT8
quantization, Core ML weight-only INT8 compression, and a search over smaller
networks. Pruning adds a different question: which parts of an already trained
network can be removed while keeping useful predictions?

## Two implementations

**Global magnitude pruning** uses PyTorch's
[`torch.nn.utils.prune.global_unstructured`](https://docs.pytorch.org/docs/stable/generated/torch.nn.utils.prune.global_unstructured.html)
with `L1Unstructured`. Across the Conv1d and Linear weight tensors, it masks the
smallest absolute weights. Biases and BatchNorm parameters are left alone. The
study tries 50% and 70% weight sparsity. Masks remain active during fine-tuning,
so removed connections cannot grow back. After choosing the best validation
epoch, `prune.remove` writes the masked weights back to ordinary parameters.
Removing the reparameterization does **not** undo pruning. The exported model
has no pruning hooks or mask buffers and uses the existing restricted checkpoint
loader.

This produces zeros inside tensors of the same shape. Ordinary dense Conv1d and
Linear kernels still process those shapes, and ordinary checkpoints still store
the tensors densely. Neither a smaller file nor a speedup should be assumed.
Compressed archive size is also not the same as deployed runtime memory. This
experiment uses dense kernels; it does not claim a sparse execution backend.

**Channel pruning** physically constructs a narrower `SensorNet` and copies only
selected channels. Within each convolution and the first linear layer, output
rows are ranked by their L1 weight magnitude. It keeps 75% or 50% of the original
width. This is an architecture-specific heuristic, not a global optimal ranking.
BatchNorm scaling and activation statistics are not part of its importance score.

Removing a convolution output also requires removing the corresponding input
channels in the next layer. The code slices the convolution weights and biases,
BatchNorm parameters and running statistics, the second convolution's input
channels, the flattened channel × time positions, and the hidden linear units.
All six output classes and all input sensor channels are preserved. A structured
zero mask on its own would not do this compaction.

The implementation supports the sequential non-fusion student architecture and
rejects unsupported fusion models. This keeps dependencies explicit for this
small network. For larger models with residual connections, concatenations, or
shared layers, use a dependency-aware graph pruning tool and verify the transformed
model rather than applying these slices blindly. No extra pruning package is
needed for the two implementations here.

## Run the comparison

Generate the original floating checkpoint and real-data arrays first:

```bash
edge-lab prepare
edge-lab train --seeds 7 --epochs 40 --compact
edge-prune --epochs 5
```

The CLI defaults to `artifacts/edge/seed-7/distilled/model.pt`. Options include
`--checkpoint`, `--data-root`, `--output`, `--report`, `--epochs`, and `--seed`.
It writes checkpoints below ignored `artifacts/pruning/` and a portable result
file to `reports/pruning.json`. Rerunning replaces this dedicated pruning report;
it does not change the original study's `reports/results.json`.

The fixed comparison contains:

- the original dense model;
- the same dense model given the same fine-tuning budget;
- 50% and 70% global magnitude pruning, followed by masked fine-tuning;
- 75%-width and 50%-width compact models, followed by fine-tuning.

Each fine-tuned candidate uses the same train/validation subjects, seed,
AdamW settings, and maximum five epochs. Validation F1 selects the epoch,
including the initial pruned state as an option. Before/after validation metrics
show whether fine-tuning recovered quality. Normalization is inherited from the
source checkpoint and is not fit on validation or test data.

The candidate rule is fixed before test evaluation: choose the smallest checkpoint
within 0.02 validation macro F1 of the original dense model, breaking size ties by
validation F1. After selecting, all candidates get a held-out test report. Test
results never feed the pruning ranks, epoch selection, or winner selection. This
is an additional experiment using an existing study checkpoint; it is not a new
blind benchmark or a multi-seed significance claim.

## What to inspect

Read the [Edge Sense notebook](../notebooks/01_edge_sense.ipynb) for the measured
comparison and the [Edge Pilot notebook](../notebooks/03_edge_pilot.ipynb) for the
validation gate. Results include parameter count, actual zero-weight fraction,
checkpoint bytes, clean and gyro-outage validation F1, test F1, and timing.
Unstructured sparsity is measured over Conv1d/Linear weights, not every parameter.

CPU batch-one timings use the existing benchmark. Batch-64 CPU and available-GPU
timings use the synchronized inference helper on the same 256 validation windows,
with warmup and float32 weights. Device inputs are resident during timing; copies
and loading are excluded. Batch-one and batch-64 numbers answer different questions
and should not be compared as if they were the same request latency.

On an NVIDIA machine, install a CUDA-enabled PyTorch build compatible with the GPU,
then rerun the study. Automatic device selection uses CUDA, then Apple MPS, then
CPU. The local study can only validate backends present on its machine. The
standalone inference command can also benchmark any compact or permanently masked
floating checkpoint:

```bash
edge-infer --checkpoint artifacts/pruning/channels_50pct/model.pt --device cuda --batch-size 64
```

Tests check that the compact model matches a reference network with the removed
activations masked, that masks survive optimizer updates, and that exported models
round-trip with the same predictions. These checks catch indexing and export
mistakes; validation and test measurements assess prediction quality.

The API behavior follows the official [PyTorch pruning tutorial](https://docs.pytorch.org/tutorials/intermediate/pruning_tutorial.html).
