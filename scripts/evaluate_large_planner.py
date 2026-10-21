"""Evaluate a larger local model with the exact same tool contract and golden suite."""

from pathlib import Path

from edge_agent_lab.local_llm import evaluate_local

evaluate_local(
    Path("data/har"),
    Path("artifacts/steward-large"),
    Path("data/models/qwen7b/qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf"),
    Path("configs/steward_golden.json"),
    threads=2,
    gpu_layers=-1,
    model_label="Qwen2.5-7B-Instruct Q4_K_M",
)
