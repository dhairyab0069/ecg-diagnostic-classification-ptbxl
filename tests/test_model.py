import torch

from ecg import CLASSES
from ecg.metrics import score
from ecg.model import ResNet1D
from ecg.transforms import augment, roll_segments, shuffle_segments


def test_output_shape():
    model = ResNet1D().eval()
    with torch.no_grad():
        out = model(torch.randn(4, 12, 1000))
    assert out.shape == (4, len(CLASSES))


def test_augment_keeps_shape():
    x = torch.randn(4, 12, 1000)
    assert augment(x).shape == x.shape


def test_shuffle_keeps_segments():
    x = torch.randn(3, 12, 1000)
    y = shuffle_segments(x, 10)
    assert y.shape == x.shape
    # every original segment should still appear somewhere in the output
    xs = x.reshape(3, 12, 10, 100).permute(0, 2, 1, 3)
    ys = y.reshape(3, 12, 10, 100).permute(0, 2, 1, 3)
    for i in range(3):
        for seg in xs[i]:
            assert any(torch.equal(seg, s) for s in ys[i])


def test_roll_keeps_segment_order():
    x = torch.arange(1000.).repeat(2, 12, 1)
    y = roll_segments(x, 10)
    assert y.shape == x.shape
    # each output segment holds exactly the values of the same input segment
    xs, ys = x.reshape(2, 12, 10, 100), y.reshape(2, 12, 10, 100)
    assert torch.equal(xs.sort(-1).values, ys.sort(-1).values)


def test_augment_with_shuffle():
    x = torch.randn(4, 12, 1000)
    assert augment(x, shuffle=10).shape == x.shape


def test_score_reads_macro_and_class():
    entry = {"macro": 0.9, "per_class": {c: 0.5 for c in CLASSES}}
    assert score(entry, "macro") == 0.9 and score(entry, "MI") == 0.5
