# Study notebooks

New to the theory? Begin with the [practical tutorial course](tutorials/README.md), which builds from sensor arrays and a toy classifier to the actual tool and selection APIs. Then return here for the measured study.

Read in this order:

1. [Portfolio summary](00_portfolio_summary.ipynb): how the three projects connect, measured highlights, and evidence limits.
2. [Edge Sense](01_edge_sense.ipynb): data splits, compression, simulation-to-real comparison, and deployment parity.
3. [Data Steward](02_data_steward.ipynb): agent ablations, strict regression results, tool traces, and failure analysis.
4. [Edge Pilot](03_edge_pilot.ipynb): constraint checks, trained candidates, infeasibility, and shadow monitoring.
5. [Post-training toolkit](04_post_training_toolkit.ipynb): pruning, dynamic/static INT8, QAT, combined optimization, and deployment verification.

All five include executed outputs for GitHub review. Their report sections load committed JSON and regenerate tables and plots
without datasets, model downloads, credentials, or Apple hardware. Optional live
inference cells in Edge Sense and Edge Pilot benchmark the current machine when
local checkpoints and data are available. The toolkit has opt-in conversion and QAT
examples; QAT runs labeled fine-tuning only when its separate switch is enabled.

From the repository root, with Python 3.11 or 3.12:

```bash
uv pip install -e '.[notebooks]'
jupyter lab notebooks/00_portfolio_summary.ipynb
```

Select the installed project environment as the notebook kernel, then **Restart
Kernel and Run All Cells**. Each notebook independently locates the repository
from its current working directory and prints the evidence SHA256. Run from the
repository root or a directory below it. Experiment reproduction commands and
hardware limitations are in [the reproduction guide](../docs/REPRODUCING.md).

## Optional local GPU inference

Edge Sense and Edge Pilot end with live inference cells. They automatically choose
CUDA, then Apple MPS, then CPU; requesting a device explicitly fails if it is
unavailable. These cells need locally generated floating checkpoints and prepared
validation arrays. They skip with setup instructions in a fresh checkout.

For a full validation pass from the shell:

```bash
edge-infer --checkpoint artifacts/edge/seed-7/distilled/model.pt --device auto --batch-size 64
# NVIDIA machine with a compatible CUDA-enabled PyTorch installation:
edge-infer --checkpoint artifacts/edge/seed-7/distilled/model.pt --device cuda --batch-size 1
```

Fresh measurements go into ignored `artifacts/inference/` files. Timings include
device synchronization, exclude input transfer, and report batch duration and
throughput separately. A CPU reference checks prediction agreement. Floating GPU
inference is separate from the study's CPU dynamic INT8 and Core ML paths.
The original CPU tables remain the original measurements. The local language
model in Data Steward uses llama.cpp GPU offload, configured separately from
PyTorch; its notebook explains that distinction.
