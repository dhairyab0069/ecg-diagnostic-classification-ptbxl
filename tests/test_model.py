import pytest
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


def test_parameter_count_matches_readme():
    assert sum(p.numel() for p in ResNet1D().parameters()) == 4_084_357  # "4.08 million" in the README


def test_receptive_field_matches_readme():
    # README: each final feature sees 1,291 samples (12.9 s); measured here from input gradients
    model = ResNet1D().eval()
    x = torch.randn(1, 12, 4000, requires_grad=True)
    features = model.layers(model.stem(x))
    features[0, :, features.size(-1) // 2].sum().backward()
    seen = (x.grad.abs().sum(1)[0] > 0).nonzero().flatten()
    assert seen.max() - seen.min() + 1 == 1291


@pytest.mark.parametrize("length", [500, 1000, 5000])
def test_any_input_length(length):
    # global pooling makes the head independent of length (500 Hz recordings are 5,000 samples)
    model = ResNet1D().eval()
    with torch.no_grad():
        assert model(torch.randn(2, 12, length)).shape == (2, len(CLASSES))


def test_every_parameter_gets_a_gradient():
    model = ResNet1D()
    model(torch.randn(4, 12, 1000)).sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


def test_eval_mode_is_deterministic():
    model = ResNet1D().eval()
    x = torch.randn(2, 12, 1000)
    with torch.no_grad():
        assert torch.equal(model(x), model(x))


def test_samples_do_not_affect_each_other_in_eval():
    # batch norm uses running stats in eval, so a record's prediction must not depend on its batch
    model = ResNet1D().eval()
    x = torch.randn(4, 12, 1000)
    with torch.no_grad():
        assert torch.allclose(model(x)[:1], model(x[:1]), atol=1e-5)
