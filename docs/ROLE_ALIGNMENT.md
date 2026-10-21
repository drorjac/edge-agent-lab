# Why these projects fit the two role families

The supplied role descriptions emphasize different layers of research engineering.
The Apple-oriented role centers on efficient multimodal models and on-device
inference. The NVIDIA-oriented role centers on tool-using agent systems, retrieval,
evaluation, and engineering workflows. This repository connects those layers
through a measurable sensor-model deployment problem.

| Role signal | Concrete implementation | Evidence to discuss |
|---|---|---|
| Train deep models on multimodal data | Accelerometer/gyroscope fusion, controlled student architecture | Model code, seed study, modality limitations |
| Model compression | Distillation and actual partial dynamic INT8; Core ML weight compression | Accuracy/size/latency tradeoffs, integer checkpoint round trip |
| On-device inference | Core ML conversion, embedded preprocessing, local runtime parity | Float/compressed packages and explicitly named backend |
| Research rigor | Subject isolation, ablations, equal-label adaptation, cluster bootstrap | Full protocol and negative outcomes |
| Data fetching and management | Pinned source hash, manifests, integrity checks, SQLite catalog | Dataset lineage and executable audits |
| Agent planning and tool use | Policy and optional local LLM planners, typed bounded executor | Complete traces, task outcomes, failure records |
| Retrieval | TF-IDF and lexical reciprocal-rank fusion over engineering evidence | Relevant-document hit@3 on golden tasks |
| Evaluation backbone | Golden tasks, held-out ML evaluation, telemetry replay | Separate model, retrieval, and workflow metrics |
| HW/SW engineering workflow | Constraint-aware candidate selection and runtime benchmarking | Critic margins, Pareto frontier, explicit infeasibility |
| Production-oriented software practices | CLI, package, tests, CI, deterministic configuration | Fresh-environment reproduction |

## How to present it

For efficient-AI interviews, lead with Edge Sense: describe the leakage risk,
show the teacher/student tradeoff, explain where compression did or did not help,
and distinguish export success from measured device execution.

For agentic-systems interviews, lead with Data Steward and Edge Pilot: show a
real task trace, explain the typed contract and evidence provenance, compare the
policy and local-model planners, and discuss why task evaluation needs more than
well-formed JSON.

For both, demonstrate one workflow from archive to selected model and incident
replay. Discuss decisions and limitations instead of presenting a collection of
framework names.

This is relevant portfolio evidence. It does not substitute for the roles'
experience requirements, production deployment history, or research publications.
