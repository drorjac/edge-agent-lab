# Model card

## Intended use

Research and engineering demonstrations of six-class activity classification,
model compression, constrained selection, and deployment parity. The intended
input is a `[1, 9, 128]` window in the primary dataset's channel order and units.
This model is not validated for medical decisions, safety monitoring, arbitrary
phone positions, or unseen wearable configurations.

## Architecture and training

The teacher has a separate convolutional stem for each three-channel stream,
then a shared temporal encoder and classifier. The compact model uses a shared
encoder with width 12. Both embed training-only channel normalization. See the
[experiment protocol](EXPERIMENTS.md) for optimizer, seed, distillation, and
adaptation settings. Checkpoints include model configuration and state.

## Compression

Distillation compares identical student architectures. Dynamic INT8 changes
Linear layers to integer kernels; Conv1d remains floating point. The checkpoint
format encodes quantized weights as integer tensors and primitive quantization
metadata so PyTorch's restricted loader can reload them without unrestricted
pickle loading. Core ML separately compresses eligible weights to INT8 with
floating-point activations.

The code does not claim pruning, hardware-aware NAS, peak-RAM reductions, or
energy savings. Parameter count, serialized storage, and latency answer different
questions and should not be substituted for one another.

## Evaluation and reliability

Validation chooses epochs and the pilot winner. The untouched official test
partition provides reporting metrics. Subject-level bootstrap intervals respect
the grouping better than window-wise bootstrapping, though only nine test people
are available. Calibration error uses ten fixed confidence bins. Stress tests
are synthetic perturbations, and the external transfer probe has substantial
modality and population mismatch.

See [measured results](../reports/RESULTS.md), including unsuccessful transfer or
compression outcomes. The selected deployment candidate is conditional on the
configured validation and machine-performance constraints; it is not a universal
best model.

## Deployment contract

Core ML models accept raw-unit sensor values and contain normalization. Python
Core ML and native Swift predictions are compared on the same saved public-data
input. The local benchmark explicitly requests CPU_ONLY. Actual Neural Engine
operator placement and mobile battery behavior remain unmeasured.
