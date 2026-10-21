# Experiment protocol

## Questions

1. How much validation and held-out accuracy survives reducing a multistream teacher to a small student?
2. Does knowledge distillation help the identical student architecture compared with supervised training alone?
3. What happens when the gyroscope disappears or sensor signals change gain, noise, or orientation?
4. Does a procedural motion prior improve real-data learning with only 480 labeled training windows?
5. Can a bounded engineering controller select a model satisfying measured resource and robustness gates?
6. Does a small local language model complete data stewardship tasks through typed tools reliably enough to improve on a policy baseline?

## Fixed data protocol

The UCI HAR archive has 10,299 preprocessed windows from 30 people. Keep its official
nine test subjects unchanged. Validation subjects are **5, 15, 19, 26**, taken from
the official training partition. Remaining training subjects provide 5,970 windows;
validation has 1,382; test has 2,947. The dataset includes 50% overlapping windows,
which is why random window splitting is inappropriate.

All nine input channels are used by default. Body acceleration and total
acceleration are related quantities, not independent sensor modalities. The true
modalities are accelerometer and gyroscope; body acceleration is a processed view.
The `accel_only` ablation uses the three body acceleration channels specifically,
so it also removes gravity information in total acceleration. Interpret it as a
restricted-input ablation rather than a perfectly controlled gyroscope ablation.

Fit channel mean and scale only on the data used to train each model. Embed those
buffers in the model. Choose the best epoch by validation macro F1. Never use test
metrics to tune an epoch, objective, simulator, or winner.

## Candidate matrix

| Candidate | Training data | Architecture | Purpose |
|---|---|---|---|
| teacher | all real train | three stems, width 32 | Larger sensor fusion reference |
| student | all real train | shared CNN, width 12 | Compact supervised baseline |
| distilled | all real train | same student | Isolate teacher supervision |
| robust_distilled | augmented real train | same student | Gain/noise augmentation and 15% batch-level gyro dropout |
| accel_only | all real train, body acceleration only | compact CNN | Restricted-input ablation |
| synthetic_fixed | procedural signals | same student | Approximate simulator without domain randomization |
| synthetic_randomized | randomized procedural signals | same student | Domain randomization under domain gap |
| fewshot_scratch | 80 real train windows per class | same student | Equal-label adaptation baseline |
| sim_to_real | procedural pretraining → same 480 real windows | same student | Measure real-data adaptation benefit or negative transfer |
| dynamic_int8 | distilled checkpoint | Linear INT8, floating Conv1d | Actual partial integer inference |

Seeds are **7, 23, 42**. Real-data training uses 40 epochs, AdamW, cosine learning
rate, batch size 128, gradient norm clipping, and training-only normalization.
Procedural pretraining uses 20 epochs. Distillation mixes hard-label cross entropy
and temperature-3 KL divergence equally, including the temperature-squared scale.
These are fixed modest baselines, not a hyperparameter search result.

## Metrics and stress conditions

Macro F1 weights six classes equally. Accuracy, confusion matrices, and 10-bin
expected calibration error accompany it. Held-out accuracy uncertainty uses 500
bootstrap draws of subjects, not individual overlapping windows. Across-seed
standard deviations describe training variation; they are not population CIs.

Stress evaluation applies a fixed seeded noise, gain, z-axis rotation, or total
gyroscope outage to real held-out recordings. These transformations may be more
or less physically plausible; they do not replace an independent hardware dataset.
The three vector streams receive the same orientation rotation.

Timing uses warmed batch-one inference, 30 warmups and 200 observations, reporting
p50 and p95. Serialized state size measures storage, not peak RAM or energy. Final
selection remeasures candidates under the then-current machine load. CPU training
and local-LLM inference can contaminate development timing; the replication guide
explains isolated timing.

## Agent evaluation

Golden cases specify a natural-language engineering task, required tools, and
relevant evidence IDs. Pass requires completion, tool coverage, and retrieval
hit@3. The policy baseline is deterministic. Optional local Qwen planning is actual
model inference with JSON grammar constraints and the same executor. Small curated
fixtures cannot establish general long-horizon or production performance.

The pilot is an adaptive policy over trained experiments, not autonomous research
invention. Its workflow coordinates data preflight, candidate measurement,
constraint critique, selection, and a synthetic incident replay. Deployment remains
a recorded selection decision; nothing is promoted into a real production system.
