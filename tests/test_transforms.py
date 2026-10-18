import pytest
import torch

from ecg.transforms import augment, roll_segments, shuffle_segments


def leads_from_one(n=3, t=1000):
    """Batch whose 12 leads are scaled copies of lead 0, so any lead mix-up is detectable."""
    base = torch.randn(n, 1, t)
    return base * torch.arange(1, 13.0).view(1, 12, 1)


def test_shuffle_moves_all_leads_together():
    y = shuffle_segments(leads_from_one(), 10)
    assert torch.allclose(y, y[:, :1] * torch.arange(1, 13.0).view(1, 12, 1))


def test_roll_moves_all_leads_together():
    y = roll_segments(leads_from_one(), 10)
    assert torch.allclose(y, y[:, :1] * torch.arange(1, 13.0).view(1, 12, 1))


@pytest.mark.parametrize("fn", [shuffle_segments, roll_segments])
def test_values_are_rearranged_not_changed(fn):
    x = torch.randn(4, 12, 1000)
    y = fn(x, 20)
    assert torch.equal(x.sort(-1).values, y.sort(-1).values)


def test_shuffle_with_one_segment_is_identity():
    x = torch.randn(2, 12, 1000)
    assert torch.equal(shuffle_segments(x, 1), x)


def test_shuffle_actually_reorders():
    torch.manual_seed(0)
    x = torch.arange(1000.).repeat(8, 12, 1)
    assert not torch.equal(shuffle_segments(x, 50), x)


def test_each_record_gets_its_own_permutation():
    torch.manual_seed(0)
    x = torch.arange(1000.).repeat(16, 12, 1)
    y = shuffle_segments(x, 10)
    assert len({tuple(row[0, ::100].tolist()) for row in y}) > 1


def test_shuffle_keeps_each_segment_intact():
    # every output segment must be one contiguous input segment, not a mix
    x = torch.arange(1000.).repeat(2, 12, 1)
    segs = shuffle_segments(x, 10).reshape(2, 12, 10, 100)
    starts = segs[..., 0]
    assert torch.equal(segs, starts[..., None] + torch.arange(100.))
    assert torch.equal(starts[0, 0].sort().values, torch.arange(0, 1000, 100.))


def test_roll_keeps_segments_in_place():
    x = torch.arange(1000.).repeat(2, 12, 1)
    segs = roll_segments(x, 10).reshape(2, 12, 10, 100)
    assert torch.equal(segs.min(-1).values[0, 0], torch.arange(0, 1000, 100.))


def test_probes_are_reproducible_with_a_seed():
    # train.py seeds before each probe so every model sees the same shuffled ECGs
    x = torch.randn(4, 12, 1000)
    for fn in (shuffle_segments, roll_segments):
        torch.manual_seed(7)
        a = fn(x, 10)
        torch.manual_seed(7)
        assert torch.equal(a, fn(x, 10))


def test_length_not_divisible_is_truncated():
    # documents current behaviour: the remainder samples are dropped
    assert shuffle_segments(torch.randn(1, 12, 1000), 3).shape == (1, 12, 999)


def test_augment_can_be_switched_off():
    x = torch.randn(4, 12, 1000)
    y = augment(x, scale_min=1.0, scale_max=1.0, noise_std=0.0, random_shift=False)
    assert torch.equal(x, y)


def test_augment_does_not_modify_its_input():
    x = torch.randn(4, 12, 1000)
    before = x.clone()
    augment(x, shuffle=10)
    assert torch.equal(x, before)


def test_augment_scale_stays_in_range():
    x = torch.ones(64, 12, 1000)
    y = augment(x, scale_min=0.8, scale_max=1.2, noise_std=0.0, random_shift=False)
    per_record = y[:, 0, 0]
    assert per_record.min() >= 0.8 and per_record.max() <= 1.2
    assert torch.equal(y, per_record.view(-1, 1, 1).expand_as(y))  # one scale per record, all leads
