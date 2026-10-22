# Optimizing a trained sensor network

The [post-training toolkit notebook](../notebooks/04_post_training_toolkit.ipynb)
collects the runnable examples and saved comparisons in one place. Start with a
trained floating-point checkpoint. Keep the original as a reference and change
one thing at a time.

## Methods in this repository

| Method | Changes weights? | Needs representative inputs? | Needs labels / optimizer? | Runtime used here |
|---|---|---|---|---|
| Dynamic INT8 | Quantizes linear weights; activation scales are dynamic | No calibration pass | No | PyTorch CPU |
| Static PTQ INT8 | Quantizes convolution and linear weights and activations | Training-only calibration windows | No | PyTorch CPU QNNPACK |
| Magnitude pruning | Zeros small individual weights | Not for ranking | Optional recovery fine-tuning uses labels | Dense CPU / floating GPU |
| Channel pruning | Removes physical channels and dependent dimensions | Not for L1 ranking | Recovery fine-tuning uses labels | Dense CPU / floating GPU |
| Channel pruning + static INT8 | Compacts first, then quantizes the compact network | Training-only calibration windows | Compaction was fine-tuned before PTQ | PyTorch CPU QNNPACK |
| QAT | Trains with simulated quantization, then converts to INT8 | Training inputs | Yes | CPU fine-tuning and CPU INT8 deployment |
| Core ML weight compression | Compresses weights, not all activations | No PTQ calibration in the existing export | No | Core ML CPU_ONLY measurements |

QAT and pruning recovery are fine-tuning extensions to the toolkit, not training-free
post-training transformations. Distillation and width search are covered in the
original model notebook; they are training methods rather than PTQ.

## Static INT8 implementation

The repository pins PyTorch 2.2.2, so the implementation uses its FX graph mode APIs:
`prepare_fx`, `get_default_qconfig_mapping`, and `convert_fx`. FX fuses supported
convolution / BatchNorm / ReLU patterns, inserts observers, and lowers supported
layers to the chosen CPU INT8 backend. The code checks the converted layer types
and weights: **both convolutions and both linear layers must be quantized**.
A model containing only fake-quantization modules does not pass that check.

“Full-network INT8” here refers to convolution and linear compute coverage. It
is not a claim that every operation is integer. Input normalization remains float,
outputs return float logits, and the FX graph can insert dequantize / requantize
boundaries around operations such as flatten. The notebook shows the actual graph
and layer inventory so these boundaries are visible.

The main calibration set contains 512 randomly selected training windows, sampled
without replacement with seed 7. A 64-window subset is a calibration-size ablation.
The selected indices, class counts, training file checksum, and source checkpoint
checksum are saved. Validation and test samples do not fit activation ranges.
Calibration runs in evaluation/inference mode without labels or optimizer steps.
Source training normalization remains embedded in the graph.

## QAT implementation

`prepare_qat_fx` inserts fake quantizers around supported operations. Observer
ranges are initialized on training windows with BatchNorm in evaluation mode.
Fine-tuning then runs for at most five epochs with AdamW at 0.0001 learning rate,
using the real training labels. BatchNorm statistics freeze after the first epoch;
observers freeze after the second. Fake quantization stays active.

Every epoch is converted to a real INT8 candidate, and its validation F1 determines
whether it replaces the current best candidate. The initialized converted model
is also eligible, so an unsuccessful fine-tuning run need not degrade the output.
A floating control receives the same optimizer, seed, and epoch budget. This avoids
attributing an effect of additional training to quantization itself.

## Deployment archives and measurements

Every candidate, including the floating controls, is traced to a TorchScript
archive and loaded back for evaluation. The archive round-trip checks raw logits
on multiple validation inputs. This is a different format from the earlier
`SensorNet` state checkpoints; the generic `load_model` helper must not be used
for these `.ts` files. Select the recorded quantized engine before loading an INT8
archive:

```python
import torch

torch.backends.quantized.engine = "qnnpack"
model = torch.jit.load("artifacts/quantization/static_ptq_512/model.ts", map_location="cpu").eval()
with torch.inference_mode():
    logits = model(windows)  # float32 CPU tensor shaped (batch, 9, 128)
    probabilities = logits.softmax(1)
```

File-size comparisons use the same deployment format for all quantization
candidates. Do not directly compare these byte counts with the pruning report's
state-checkpoint counts. Packed weights, graph metadata, and archive structure
all contribute to file size.

Timing measures the loaded TorchScript model, not an unexported Python model.
Three interleaved rounds with shuffled candidate order report median p50/p95 and
per-round p95 ranges. Each measurement has warmup and uses resident CPU tensors.
Batch-one latency and batch-64 duration are reported separately. File loading,
softmax, data transfer, and an application server are outside the timer. A smaller
archive or lower arithmetic precision does not guarantee lower measured latency.

The converted QNNPACK INT8 graphs are CPU artifacts. CUDA availability does not
make them CUDA kernels. Floating original and compact models can use the separate
`edge-infer` CUDA/MPS benchmark. NVIDIA INT8 deployment would need a distinct
supported runtime/export path, such as TensorRT, plus verification on NVIDIA
hardware; it is not claimed by this experiment.

## Reproduce

With the project environment active and the original student checkpoint available:

```bash
edge-quantize --epochs 5
```

For the combined channel-pruning + PTQ comparison:

```bash
edge-prune --epochs 5
edge-quantize --epochs 5 --compact-checkpoint artifacts/pruning/channels_50pct/model.pt
```

The quantization CLI supports `--checkpoint`, `--compact-checkpoint`, `--data-root`,
`--output`, `--report`, `--epochs`, `--seed`, and `--backend`. QNNPACK is used for the
recorded run. A backend not available in the installed PyTorch build raises an
error. Do not compare results from different backends as if the environment were
unchanged.

Calibration ranges and fine-tuning choices are fixed before the test set is loaded.
The selection rule is the smallest deployment archive whose validation F1 stays
within 0.02 of the original float model. Test scores are reporting only. One
checkpoint and one fine-tuning seed provide an engineering comparison, not a
significance result or a target-device guarantee.

## API version and sources

This is the API path exercised by the pinned environment, rather than a claim that
FX/TorchScript is the current preferred starting point for a new PyTorch project.
Current PyTorch documentation directs newer quantization development toward
**torchao / PT2E**. Migrating this project requires a separate dependency environment
and rechecking Conv1d coverage, export, and backend behavior; those APIs should not
be dropped into PyTorch 2.2.2 unchanged.

- [PyTorch FX preparation API](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_fx.prepare_fx.html)
- [FX conversion API](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_fx.convert_fx.html)
- [QAT preparation API](https://docs.pytorch.org/docs/main/generated/torch.ao.quantization.quantize_fx.prepare_qat_fx.html)
- [Current quantization migration guidance](https://docs.pytorch.org/docs/main/quantization)
- [Pruning implementation](PRUNING.md)
