import numpy as np
import pytest
from scipy.io import savemat

from ecg import CLASSES
from ecg.external import LEADS, SINUS_RHYTHM, labels_from_dx, load_record, parse_header, to_100hz

HEADER = """E00001 12 500 5000
""" + "".join(f"E00001.mat 16x1+24 1000.0(0)/mV 16 0 136 -28477 0 {lead}\n" for lead in LEADS) + """# Age: 63
# Sex: Female
# Dx: 426783006,164873001
# Rx: Unknown
"""


def onehot(*classes):
    return np.array([c in classes for c in CLASSES], dtype=np.float32)


def test_parse_header():
    fs, n, leads, gains, dx = parse_header(HEADER)
    assert (fs, n) == (500, 5000) and leads == LEADS
    assert gains == [1000.0] * 12 and dx == ["426783006", "164873001"]


def test_sinus_rhythm_alone_is_norm():
    assert np.array_equal(labels_from_dx([SINUS_RHYTHM]), onehot("NORM"))


def test_sinus_rhythm_with_a_finding_is_not_norm():
    assert np.array_equal(labels_from_dx([SINUS_RHYTHM, "164873001"]), onehot("HYP"))


def test_several_classes():
    assert np.array_equal(labels_from_dx(["59118001", "425623009", "164873001"]), onehot("CD", "STTC", "HYP"))


@pytest.mark.parametrize("dx", [["164934002"], ["164889003"], [SINUS_RHYTHM, "164917005"], []])
def test_form_and_rhythm_only_records_are_dropped(dx):
    # T-wave abnormal, AF, and Q wave abnormal are form/rhythm statements in PTB-XL, so no label
    assert labels_from_dx(dx) is None


def test_load_record_converts_to_mv(tmp_path):
    raw = (np.arange(12 * 5000).reshape(12, 5000) % 2000 - 1000).astype(np.int16)
    savemat(tmp_path / "r.mat", {"val": raw})
    x = load_record(tmp_path / "r.mat", [1000.0] * 12)
    assert x.shape == (5000, 12) and np.allclose(x, raw.T / 1000)


def test_resample_500_to_100_hz_keeps_slow_signals():
    t = np.arange(5000) / 500
    x = np.repeat(np.sin(2 * np.pi * 1.0 * t)[:, None], 12, axis=1)  # 1 Hz, well below the new Nyquist
    y = to_100hz(x, 500)
    assert y.shape == (1000, 12)
    assert np.allclose(y[50:-50, 0], np.sin(2 * np.pi * np.arange(1000) / 100)[50:-50], atol=0.01)
