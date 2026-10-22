# Pruning results

Single source checkpoint and fine-tuning seed 7. Ratios and the validation selection rule were fixed before held-out test evaluation. See [method and reproduction](../docs/PRUNING.md).

| Candidate | Parameters | Zero weights | Checkpoint KiB | Val before tuning | Val after tuning | Test F1 | CPU batch-1 p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| dense_original | 13,662 | 0% | 59.6 | 0.9602 | 0.9602 | 0.9139 | 0.088 |
| dense_finetuned | 13,662 | 0% | 59.6 | 0.9602 | 0.9602 | 0.9139 | 0.088 |
| magnitude_50 | 13,662 | 50% | 59.6 | 0.9485 | 0.9553 | 0.9209 | 0.089 |
| magnitude_70 | 13,662 | 70% | 59.6 | 0.7342 | 0.9531 | 0.8832 | 0.088 |
| channels_75pct | 7,980 | 0% | 37.7 | 0.5532 | 0.9672 | 0.8997 | 0.087 |
| channels_50pct | 3,810 | 0% | 20.8 | 0.2901 | 0.9450 | 0.8969 | 0.087 |

Selected by the validation-only size/quality rule: **channels_50pct**. Minimum validation F1: 0.9402.

Timing is the median of three interleaved post-training rounds. Batch-64 CPU and available-device measurements, timing ranges, and fine-tuning histories are in [pruning.json](pruning.json). These are host-specific measurements, not a sparse-kernel acceleration claim.

Magnitude pruning leaves dense tensor dimensions and stored checkpoint size unchanged. Physical channel compaction reduces parameter count and file size. The half-width candidate passes the validation rule but has lower held-out F1 than the original. One seed is insufficient to establish a stable accuracy gain or ranking.
