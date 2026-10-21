import numpy as np
import pytest

from edge_agent_lab.data import Split, assert_disjoint, corrupt, normalization, simulate


def test_subject_leakage_rejected():
    a = Split(np.zeros((2, 9, 128)), np.array([0, 1]), np.array([1, 2]))
    b = Split(np.zeros((2, 9, 128)), np.array([0, 1]), np.array([2, 3]))
    with pytest.raises(ValueError, match="leakage"):
        assert_disjoint({"train": a, "test": b})


def test_simulation_is_reproducible_balanced_and_distinguishes_domain_randomization():
    fixed, repeated = simulate(120, 7, False), simulate(120, 7, False)
    randomized = simulate(120, 7, True)
    np.testing.assert_array_equal(fixed.x, repeated.x)
    assert fixed.x.shape == (120, 9, 128)
    assert np.isfinite(randomized.x).all()
    assert np.bincount(randomized.y).tolist() == [20] * 6
    assert not np.allclose(fixed.x, randomized.x)
    assert (randomized.subjects == -1).all()


def test_dropout_targets_only_gyroscope_without_mutation():
    x = np.ones((4, 9, 128), dtype="float32")
    out = corrupt(x, "gyro_dropout")
    assert not out[:, 3:6].any()
    assert out[:, :3].all() and out[:, 6:].all() and x.all()


def test_rotation_preserves_vector_norm():
    x = np.random.default_rng(7).normal(size=(8, 9, 128)).astype("float32")
    out = corrupt(x, "rotation")
    for start in (0, 3, 6):
        np.testing.assert_allclose(
            np.linalg.norm(x[:, start : start + 3], axis=1),
            np.linalg.norm(out[:, start : start + 3], axis=1),
            rtol=1e-6,
        )


def test_normalization_handles_constant_channels():
    x = np.ones((6, 9, 128), dtype="float32")
    mean, scale = normalization(x)
    assert np.isfinite((x - mean) / scale).all()
    assert (scale > 0).all()
