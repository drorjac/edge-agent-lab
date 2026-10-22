import numpy as np
import pytest
import torch

from edge_agent_lab.common import seed_everything
from edge_agent_lab.data import simulate
from edge_agent_lab.models import SensorNet
from edge_agent_lab.quantization import (
    assert_static_coverage,
    calibration_indices,
    deployment_archive,
    prepare_static,
    set_backend,
    tune,
)


def test_static_int8_covers_convs_and_linears_and_archive_runs(tmp_path):
    seed_everything(7, threads=1)
    model = SensorNet(width=4).eval()
    data = simulate(24, seed=5)
    quantized = prepare_static(model, data.x)
    layers = assert_static_coverage(quantized)
    assert len(layers) == 4
    assert all(layer["weight_dtype"] == "torch.qint8" for layer in layers)
    restored = deployment_archive(quantized, data.x, tmp_path / "model.ts")
    with torch.inference_mode():
        actual = restored(torch.from_numpy(data.x[:3]))
        expected = quantized(torch.from_numpy(data.x[:3]))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3, 6)
    assert "quantized::conv1d" in str(restored.inlined_graph)
    assert "quantized::linear" in str(restored.inlined_graph)
    assert torch.isfinite(actual).all()
    assert model.encoder[0].weight.dtype == torch.float32


def test_calibration_sampling_is_reproducible_and_without_replacement():
    a = calibration_indices(100, 32, 7)
    np.testing.assert_array_equal(a, calibration_indices(100, 32, 7))
    assert len(set(a.tolist())) == 32
    assert len(calibration_indices(10, 32, 7)) == 10
    with pytest.raises(ValueError, match="positive"):
        calibration_indices(0, 5, 7)
    with pytest.raises(ValueError, match="unavailable"):
        set_backend("none")


def test_qat_converts_to_real_int8_after_training():
    from torch.ao.quantization import get_default_qat_qconfig_mapping
    from torch.ao.quantization.quantize_fx import prepare_qat_fx

    seed_everything(8, threads=1)
    set_backend("qnnpack")
    train, val = simulate(24, 5), simulate(12, 6)
    prepared = prepare_qat_fx(
        SensorNet(width=4).train(),
        get_default_qat_qconfig_mapping("qnnpack"),
        (torch.from_numpy(train.x[:1]),),
    )
    prepared.eval()
    with torch.no_grad():
        prepared(torch.from_numpy(train.x))
    converted, history = tune(prepared, train, val, epochs=1, seed=8, qat=True)
    assert_static_coverage(converted)
    assert len(history) == 2
    assert history[0]["epoch"] == 0
    assert not any("FakeQuant" in type(m).__name__ for m in converted.modules())
