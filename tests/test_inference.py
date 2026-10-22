import numpy as np
import pytest
import torch

from edge_agent_lab.inference import resolve_device, run_inference
from edge_agent_lab.models import SensorNet


def test_cpu_prediction_matches_direct_forward():
    torch.set_num_threads(1)
    torch.manual_seed(4)
    model = SensorNet(width=4).eval()
    x = np.random.default_rng(4).normal(size=(5, 9, 128)).astype(np.float32)
    with torch.inference_mode():
        expected = model(torch.from_numpy(x)).softmax(1).numpy()
    actual, timing = run_inference(model, x, "cpu", batch_size=3, repeats=2, warmup=1)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    assert timing["batch_size"] == 3
    assert timing["batch_p95_ms"] > 0
    assert timing["device"] == "cpu"
    assert next(model.parameters()).device.type == "cpu"


def test_explicit_unavailable_device_does_not_silently_fallback(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device().type == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is unavailable"):
        resolve_device("cuda")
    with pytest.raises(RuntimeError, match="MPS is unavailable"):
        resolve_device("mps")


def test_invalid_input_and_quantized_model():
    model = SensorNet(width=4).eval()
    with pytest.raises(ValueError, match="shape"):
        run_inference(model, np.empty((0, 9, 128), np.float32))
    with pytest.raises(ValueError, match="positive"):
        run_inference(model, np.zeros((1, 9, 128), np.float32), repeats=0)
    torch.backends.quantized.engine = "qnnpack"
    quantized = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    with pytest.raises(ValueError, match="CPU-only"):
        run_inference(quantized, np.zeros((1, 9, 128), np.float32))


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_accelerator_prediction_agrees_with_cpu(device):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} unavailable")
    torch.manual_seed(5)
    model = SensorNet(width=4).eval()
    x = np.random.default_rng(5).normal(size=(4, 9, 128)).astype(np.float32)
    cpu, _ = run_inference(model, x, "cpu", repeats=2, warmup=1)
    gpu, timing = run_inference(model, x, device, repeats=2, warmup=1)
    np.testing.assert_allclose(gpu, cpu, atol=2e-4, rtol=2e-4)
    assert timing["device"] == device
