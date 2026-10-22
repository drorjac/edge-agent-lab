import copy

import numpy as np
import pytest
import torch

from edge_agent_lab.common import seed_everything
from edge_agent_lab.data import simulate
from edge_agent_lab.metrics import predict
from edge_agent_lab.models import SensorNet, load_model, save_model
from edge_agent_lab.pruning import (
    apply_magnitude_pruning,
    compact_channels,
    finetune,
    remove_masks,
    weight_modules,
    weight_sparsity,
)


def test_compacted_network_matches_masked_reference_and_roundtrips(tmp_path):
    seed_everything(7, threads=1)
    source = SensorNet(width=8).eval()
    # Exercise BN slicing with nondefault buffers as well as weights and bias.
    with torch.no_grad():
        for bn in [source.encoder[1], source.encoder[4]]:
            bn.running_mean.copy_(torch.randn_like(bn.running_mean))
            bn.running_var.copy_(torch.rand_like(bn.running_var) + 0.2)
            bn.bias.copy_(torch.randn_like(bn.bias))
    compact, indices = compact_channels(source, 4)
    reference = copy.deepcopy(source)
    hooks = []
    for module, keep in [
        (reference.encoder[2], indices["first_channels"]),
        (reference.encoder[5], indices["second_channels"]),
        (reference.head[1], indices["hidden_units"]),
    ]:

        def mask_outputs(module, inputs, output, keep=keep):
            mask = torch.zeros(output.shape[1], device=output.device)
            mask[keep] = 1
            return output * mask.reshape(1, -1, *([1] * (output.ndim - 2)))

        hooks.append(module.register_forward_hook(mask_outputs))
    x = simulate(12, seed=8).x
    np.testing.assert_allclose(predict(compact, x), predict(reference, x), atol=1e-6)
    assert sum(p.numel() for p in compact.parameters()) < sum(
        p.numel() for p in source.parameters()
    )
    save_model(compact, tmp_path / "model.pt", compact.config())
    restored = load_model(tmp_path / "model.pt")
    np.testing.assert_allclose(predict(compact, x), predict(restored, x))
    for hook in hooks:
        hook.remove()


def test_masks_survive_training_and_removal(tmp_path):
    seed_everything(9, threads=1)
    model = SensorNet(width=4).eval()
    apply_magnitude_pruning(model, 0.5)
    masks = [m.weight_mask.clone() for m in weight_modules(model)]
    train, val = simulate(30, 3), simulate(12, 4)
    finetune(model, train, val, epochs=2, seed=9)
    for module, mask in zip(weight_modules(model), masks, strict=True):
        assert torch.equal(module.weight_mask, mask)
        assert torch.count_nonzero(module.weight[mask == 0]) == 0
    before = predict(model, val.x)
    remove_masks(model)
    np.testing.assert_allclose(predict(model, val.x), before)
    assert abs(weight_sparsity(model) - 0.5) < 0.001
    assert not any("mask" in key or "orig" in key for key in model.state_dict())
    save_model(model, tmp_path / "model.pt", model.config())
    np.testing.assert_allclose(predict(load_model(tmp_path / "model.pt"), val.x), before)


def test_unsupported_compaction_is_explicit():
    with pytest.raises(ValueError, match="non-fusion"):
        compact_channels(SensorNet(width=4, fusion=True), 2)
    with pytest.raises(ValueError, match="smaller"):
        compact_channels(SensorNet(width=4), 4)
    with pytest.raises(ValueError, match="between"):
        apply_magnitude_pruning(SensorNet(width=4), 1)
