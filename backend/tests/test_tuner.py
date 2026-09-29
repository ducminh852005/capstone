import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from _common import TunerBinding, draw_shuttle_trajectory  # noqa: E402
from core import config  # noqa: E402

import numpy as np  # noqa: E402


def test_missing_attribute_raises_instead_of_creating_it():
    obj = SimpleNamespace(rest_speed=2.0)
    with pytest.raises(AttributeError):
        TunerBinding("Rest", obj, "rest_speed_px", 0.0, 10.0, 0.5)
    assert not hasattr(obj, "rest_speed_px")


def test_invalid_range_rejected():
    obj = SimpleNamespace(x=1.0)
    with pytest.raises(ValueError):
        TunerBinding("X", obj, "x", 5.0, 5.0, 1.0)
    with pytest.raises(ValueError):
        TunerBinding("X", obj, "x", 0.0, 5.0, 0.0)


def test_float_binding_roundtrip():
    obj = SimpleNamespace(x=12.0)
    b = TunerBinding("X", obj, "x", 5.0, 40.0, 1.0)
    assert b.initial_position() == 7          # (12 - 5) / 1
    assert b.steps == 35
    assert b.set_position(10) == 15.0
    assert obj.x == 15.0
    assert b.text() == "X: 15.00"


def test_int_binding_stays_int_and_clamps_initial_position():
    obj = SimpleNamespace(n=1000)
    b = TunerBinding("N", obj, "n", 0, 100, 5)
    assert b.initial_position() == b.steps    # clamped to the slider end
    b.set_position(4)
    assert obj.n == 20 and isinstance(obj.n, int)
    assert b.text() == "N: 20"


def test_binding_on_module_reads_current_value(monkeypatch):
    monkeypatch.setattr(config, "SMASH_SPEED_THRESHOLD", 12.0)
    b = TunerBinding("Speed", config, "SMASH_SPEED_THRESHOLD", 5.0, 40.0, 1.0)
    b.set_position(0)
    assert config.SMASH_SPEED_THRESHOLD == 5.0


def test_trajectory_tail_length_read_at_call_time(monkeypatch):
    frame = np.zeros((50, 50, 3), np.uint8)
    traj = [(i, i) for i in range(1, 40)]
    monkeypatch.setattr(config, "TRAJECTORY_TAIL_LENGTH", 3)
    short = frame.copy()
    draw_shuttle_trajectory(short, traj)
    monkeypatch.setattr(config, "TRAJECTORY_TAIL_LENGTH", 30)
    long = frame.copy()
    draw_shuttle_trajectory(long, traj)
    assert (long > 0).sum() > (short > 0).sum()
