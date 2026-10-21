import numpy as np
import torch

from edge_agent_lab.common import seed_everything
from edge_agent_lab.data import simulate
from edge_agent_lab.metrics import classification, predict
from edge_agent_lab.models import SensorNet, load_model, save_model
from edge_agent_lab.training import train_model


def test_training_checkpoint_roundtrip_and_quantized_inference(tmp_path):
    seed_everything(3, threads=1)
    train, val = simulate(60, 1), simulate(24, 2)
    result = train_model(SensorNet(width=4), train, val, tmp_path, seed=3, epochs=2)
    restored = load_model(tmp_path / "model.pt")
    probs = predict(restored, val.x)
    assert probs.shape == (24, 6)
    np.testing.assert_allclose(probs.sum(1), 1, atol=1e-6)
    assert result["train_source"] == "procedural"
    torch.backends.quantized.engine = "qnnpack"
    quantized = torch.ao.quantization.quantize_dynamic(
        restored, {torch.nn.Linear}, dtype=torch.qint8
    )
    save_model(quantized, tmp_path / "int8.pt", restored.config())
    qloaded = load_model(tmp_path / "int8.pt", quantized=True)
    np.testing.assert_allclose(predict(quantized, val.x), predict(qloaded, val.x))


def test_metrics_do_not_drop_absent_classes():
    result = classification(np.array([0, 0]), np.array([[1, 0, 0, 0, 0, 0]] * 2))
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1 / 6
    assert result["ece_10bin"] == 0
