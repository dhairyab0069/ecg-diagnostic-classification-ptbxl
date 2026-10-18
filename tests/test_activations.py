import pytest
import torch
import torch.nn as nn

from ecg.model import ResNet1D, build_model, make_act

BASE = {"widths": [64, 128, 256, 256], "kernel_size": 7, "dropout": 0.3}
N_ACTS = 17  # stem + 2 per residual block x 8 blocks


def count(model, cls):
    return sum(type(m) is cls for m in model.modules())


@pytest.mark.parametrize("act, cls", [("relu", nn.ReLU), ("leaky_relu", nn.LeakyReLU), ("tanh", nn.Tanh)])
def test_every_activation_is_swapped(act, cls):
    model = build_model({**BASE, "activation": act})
    assert count(model, cls) == N_ACTS
    others = {nn.ReLU, nn.LeakyReLU, nn.Tanh} - {cls}
    assert all(count(model, o) == 0 for o in others)


def test_config_without_activation_builds_relu():
    # configs and checkpoints from before the activation option must still load as ReLU
    model = build_model(BASE)
    assert count(model, nn.ReLU) == N_ACTS


def test_negative_slope_is_used():
    model = build_model({**BASE, "activation": "leaky_relu", "negative_slope": 0.2})
    slopes = {m.negative_slope for m in model.modules() if isinstance(m, nn.LeakyReLU)}
    assert slopes == {0.2}


def test_unknown_activation_raises():
    with pytest.raises(ValueError, match="unknown activation"):
        make_act("gelu")


def test_activations_compute_what_they_should():
    x = torch.tensor([-2.0, -0.5, 0.0, 1.5])
    assert torch.equal(make_act("relu")(x.clone()), torch.tensor([0.0, 0.0, 0.0, 1.5]))
    assert torch.allclose(make_act("leaky_relu", 0.1)(x.clone()), torch.tensor([-0.2, -0.05, 0.0, 1.5]))
    assert torch.allclose(make_act("tanh")(x), torch.tanh(x))


@pytest.mark.parametrize("act", ["relu", "leaky_relu", "tanh"])
def test_same_parameter_count_for_every_activation(act):
    # activations have no weights, so the three models differ only in the non-linearity
    n = sum(p.numel() for p in ResNet1D(act=act).parameters())
    assert n == sum(p.numel() for p in ResNet1D().parameters())
