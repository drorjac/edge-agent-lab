# From zero to the working application

These five practical lessons teach the ideas behind Edge Agent Lab before asking
you to interpret the research notebooks. Start with basic Python and sensor arrays,
then build toward a reviewable model-selection application. No prior ML or agent
experience is assumed. Basic comfort editing and running Python cells is helpful;
the first lesson introduces the Python containers and array conventions used here.

## Learning path

| Lesson | Theory | Practical outcome |
|---|---|---|
| [1. Data and Python](01_data_and_python.ipynb) | Windows, channels, splits, normalization, fingerprints | Inspect arrays, detect subject leakage, identify real evidence |
| [2. Learning and evaluation](02_learning_and_evaluation.ipynb) | Scores, softmax, cross-entropy, gradients, macro F1 | Train a toy classifier and run the actual SensorNet interface |
| [3. Optimization](03_optimization.ipynb) | Distillation, pruning, INT8, calibration, fair measurement | Demonstrate masking and rounding, then inspect actual deployment measurements |
| [4. Tools and agents](04_tools_and_agents.ipynb) | Retrieval, typed actions, execution, observations | Search project runbooks, reject unsupported tools, inspect a saved investigation |
| [5. Combined application](05_combined_application.ipynb) | Hard gates, ranking, infeasibility, provenance | Recompute decisions with the actual selection functions and create an in-memory decision artifact |

Each lesson has theory before code, questions to consider before running, exercises,
worked answers, and a bridge to the implementation. Allow approximately 2–4 hours
including exercises. Synthetic examples are explicitly labeled. Saved project
measurements remain separate from new teaching calculations.

## Setup

Use Python 3.11 or 3.12 and the existing project environment. From the repository root:

```bash
source .venv/bin/activate
uv pip install -e '.[notebooks]'
jupyter lab notebooks/tutorials/01_data_and_python.ipynb
```

If you have no environment yet, first follow the environment creation steps in
[the main README](../../README.md#quick-start). Select the project Python environment
as your kernel. Run each lesson from top to bottom with a fresh kernel. All five
lessons run independently: knowledge builds between lessons, but Python variables
do not carry between notebooks.

The default lessons run on CPU without downloads, checkpoints, paid APIs, or
credentials. They use installed project dependencies and committed JSON reports.
They do not launch full training, write deployment files, or replace study reports.
The final lesson lists the longer real-data workflow as explicit shell commands.

## Check your understanding

Before moving on, explain these points in your own words:

- Why are people, rather than individual overlapping windows, separated across splits?
- Why can high accuracy conceal a failed activity class?
- Why can 50% zero weights leave dense storage almost unchanged?
- Why can a valid tool request still investigate the wrong incident?
- Why must the controller allow a no-feasible-candidate outcome?

Next, follow the [study notebooks](../README.md) for measured experiments and
limitations, and the [reproduction guide](../../docs/REPRODUCING.md) to run the full
application on real data.
