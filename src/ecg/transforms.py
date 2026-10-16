"""Training augmentation and the temporal shuffle probe."""
import torch


def augment(x, scale_min=0.8, scale_max=1.2, noise_std=0.05, random_shift=True, shuffle=0):
    if shuffle:  # train on shuffled ECGs (the shuffle-trained variant)
        x = shuffle_segments(x, shuffle)
    scale = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(scale_min, scale_max)
    x = x * scale
    if random_shift:
        x = torch.roll(x, int(torch.randint(0, x.size(-1), (1,))), dims=-1)
    return x + noise_std * torch.randn_like(x)


def _split(x, num_segments):
    n, c, t = x.shape
    seg = t // num_segments
    return x[..., : seg * num_segments].reshape(n, c, num_segments, seg)


def shuffle_segments(x, num_segments=10):
    """Split each ECG into equal segments and randomly permute them.

    Keeps local beat shape, destroys rhythm and temporal order.
    """
    n, c, _ = x.shape
    x = _split(x, num_segments)
    idx = torch.argsort(torch.rand(n, num_segments, device=x.device), dim=1)
    x = x[torch.arange(n, device=x.device)[:, None], :, idx]  # (n, k, c, seg)
    return x.permute(0, 2, 1, 3).reshape(n, c, -1)


def roll_segments(x, num_segments=10):
    """Control for the shuffle probe: cut at the same boundaries but keep segment order.

    Each segment is circularly shifted by a random offset, so the joins between
    segments become discontinuous as in the shuffle, while the order of segments
    (and the coarse rhythm) is kept.
    """
    n, c, _ = x.shape
    x = _split(x, num_segments)
    seg = x.size(-1)
    shift = torch.randint(0, seg, (n, 1, num_segments, 1), device=x.device)
    pos = (torch.arange(seg, device=x.device) + shift) % seg
    return torch.gather(x, -1, pos.expand(n, c, num_segments, seg)).reshape(n, c, -1)
