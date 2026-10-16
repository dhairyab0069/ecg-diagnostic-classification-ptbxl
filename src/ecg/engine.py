"""Training and inference loops."""
import torch

from ecg.metrics import auroc
from ecg.transforms import augment


def train_one_epoch(model, loader, opt, sched, scaler, loss_fn, device, aug_cfg):
    model.train()
    total = 0.0
    amp = device.type == "cuda"
    for x, y in loader:
        x = augment(x.to(device, non_blocking=True), **aug_cfg)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=amp):
            loss = loss_fn(model(x), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        sched.step()
        total += loss.item()
    return total / len(loader)


@torch.no_grad()
def predict(model, loader, device, transform=None):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        if transform is not None:
            x = transform(x)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(x.to(device, non_blocking=True))
        probs.append(torch.sigmoid(logits.float()).cpu())
        targets.append(y)
    return torch.cat(probs).numpy(), torch.cat(targets).numpy()


def evaluate(model, loader, device, transform=None):
    """Return {"macro": AUROC, "per_class": {class: AUROC}} on a loader."""
    macro, per_class = auroc(*predict(model, loader, device, transform))
    return {"macro": macro, "per_class": per_class}
