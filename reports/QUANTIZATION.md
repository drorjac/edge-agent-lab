# Quantization results

All sizes and timings below use loaded TorchScript deployment archives. CPU backend: QNNPACK, PyTorch 2.2.2, four threads. One source checkpoint and seed 7. See [implementation and reproduction](../docs/POST_TRAINING.md).

| Candidate | Archive KiB | Validation F1 | Test F1 | CPU batch-1 p95 ms | CPU batch-64 p95 ms |
|---|---:|---:|---:|---:|---:|
| float_original | 78.5 | 0.9602 | 0.9139 | 0.066 | 3.440 |
| float_finetuned | 79.4 | 0.9602 | 0.9139 | 0.066 | 3.448 |
| dynamic_int8_linear | 56.7 | 0.9602 | 0.9139 | 0.146 | 3.462 |
| static_ptq_64 | 39.7 | 0.9575 | 0.9143 | 0.060 | 0.602 |
| static_ptq_512 | 40.2 | 0.9582 | 0.9137 | 0.060 | 0.616 |
| qat_int8 | 40.2 | 0.9575 | 0.9129 | 0.060 | 0.604 |
| compact_float | 41.0 | 0.9450 | 0.8969 | 0.056 | 2.021 |
| compact_static_ptq_512 | 30.6 | 0.9462 | 0.8989 | 0.050 | 0.474 |

Static PTQ and QAT have real INT8 weights and kernels in both convolutions and both linear layers. Normalization, output logits, and some graph boundaries remain float. Dynamic INT8 covers the linear layers only.

QAT and the extra-trained float control selected their initialized epoch-zero candidates: the five fine-tuning epochs did not improve the actual deployment validation score. This is a negative fine-tuning result, not a claimed QAT improvement.

Validation-only selection chose **compact_static_ptq_512** under the preset quality floor 0.9402. The compact candidate trades held-out quality for size. Test metrics were not used to select candidates.

Timings summarize three interleaved rounds; raw samples and per-round ranges are in [quantization.json](quantization.json). Archive sizes include graph and packing metadata and should not be directly compared with the pruning report’s state checkpoints. See the [converted graph](static_int8_graph.txt) for quantize/dequantize boundaries.
