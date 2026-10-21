# Engineering decisions

- Use one installable monorepo with three independently runnable projects. Shared
  ingestion and metrics prevent three slightly different evaluation protocols.
- Use UCI HAR accelerometer and gyroscope recordings: sensor fusion fits edge ML
  and supports a credible engineering-agent use case without enormous GPU costs.
- Preserve the official subject-held-out test split. Select validation subjects
  from training subjects only. Fit preprocessing on the remaining training data.
- Distinguish a procedural motion simulator, synthetic corruptions of real
  signals, and actual measured recordings in every report.
- Default agent planning is deterministic and inspectable. An optional local LLM
  planner can propose typed actions, but cannot execute arbitrary code or SQL.
- Python 3.12 and PyTorch 2.2.2 support this machine's macOS 13.4 environment.
  Core ML export and runtime results must be reported separately.
- Save observed results, failed hypotheses, environment metadata, and limitations.
  A small honest benchmark is preferable to an unverified performance claim.

- The primary test, external transfer probe, and agent regression suite answer
  different questions. Their scores are never merged into one "success" number.
- On-demand search uses an allowlisted architecture space and real training tools
  under a tighter storage objective. It is a bounded policy, not free-form NAS.
- Schema-only and workflow-guarded local LLM runs retain semantic failures. A
  separate policy-fixed realization ablation demonstrates correctness inherited
  from explicit policy and its added inference cost.
- Large-model inference is scheduled separately after observed unified-memory
  pressure. Evaluation saves per-case results and resumes without rerunning
  completed cases. A model file hash is calculated once per evaluation process.
- Publish code and portable evidence to a new private personal repository. Preserve
  raw data and model caches locally in ignored directories.
