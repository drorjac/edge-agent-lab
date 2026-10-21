# Contributing

Install the development extras, then run:

```bash
ruff check src tests scripts
ruff format --check src tests scripts
pytest -q
```

Changes to data splits, input preprocessing, model selection, or objective
semantics need focused regression tests and an update to the experiment protocol.
Keep real-data evaluation out of CI. Add failure cases when extending agent tools;
well-formed actions alone are not evidence of task success.

Do not commit raw datasets, large model weights, credentials, or private traces.
Keep measured results separate from expectations and label simulated data
explicitly. If the experiment protocol changes, write results to a new artifact
directory rather than reusing incompatible checkpoints.
