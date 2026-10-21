import numpy as np
import pytest

from edge_agent_lab.external import bridge_recording


def signals(n=256):
    total = np.zeros((n, 3))
    total[:, 0] = -1
    labels = np.full(n, 6, dtype="int64")
    timestamps = np.datetime64("2021-01-01", "ms") + np.arange(n) * np.timedelta64(20, "ms")
    return total, labels, timestamps


def test_external_bridge_keeps_real_gravity_and_marks_absent_gyro():
    total, labels, timestamps = signals()
    x, y, stats = bridge_recording(total, labels, timestamps)
    assert x.shape == (2, 9, 128)
    assert y.tolist() == [4, 4]
    np.testing.assert_allclose(x[:, :3], 0, atol=1e-5)
    assert not x[:, 3:6].any()
    np.testing.assert_allclose(x[:, 6], -1)
    assert stats["retained_windows"] == 2


def test_external_bridge_drops_unmapped_or_transition_windows():
    total, labels, timestamps = signals()
    labels[:128] = 3  # shuffling is not silently mapped to walking
    labels[200] = 1  # a mixed-label window is not assigned a convenient majority label
    x, y, stats = bridge_recording(total, labels, timestamps)
    assert len(x) == len(y) == 0
    assert stats["transition_or_unmapped_windows"] == 2


def test_external_windows_never_bridge_time_gaps():
    total, labels, timestamps = signals()
    timestamps[64:] += np.timedelta64(10, "s")
    x, _, stats = bridge_recording(total, labels, timestamps)
    assert len(x) == 1
    assert stats["time_discontinuities"] == 1


def test_external_rejects_nonfinite_recordings():
    total, labels, timestamps = signals()
    total[0, 0] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        bridge_recording(total, labels, timestamps)
