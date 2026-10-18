import numpy as np
import pytest
import torch
import yaml

from ecg import CLASSES
from ecg.data import build_splits, load_ptbxl, normalize, record_column
from ecg.metrics import auroc
from ecg.utils import load_config


def data_cfg(root):
    return {"root": str(root), "sampling_rate": 100, "train_folds": [1, 2, 3, 4, 5, 6, 7, 8],
            "val_fold": 9, "test_fold": 10}


def test_record_column():
    assert record_column(100) == "filename_lr" and record_column(500) == "filename_hr"


def test_load_reads_the_cache(fake_ptbxl):
    root, (X, Y, fold) = fake_ptbxl
    X2, Y2, fold2 = load_ptbxl(str(root))
    assert np.array_equal(X, X2) and np.array_equal(Y, Y2) and np.array_equal(fold, fold2)


def test_splits_follow_the_folds(fake_ptbxl):
    root, (_, _, fold) = fake_ptbxl
    splits, _ = build_splits(data_cfg(root))
    assert len(splits["train"][0]) == np.isin(fold, range(1, 9)).sum()
    assert len(splits["val"][0]) == (fold == 9).sum()
    assert len(splits["test"][0]) == (fold == 10).sum()


def test_splits_do_not_overlap(fake_ptbxl):
    root, _ = fake_ptbxl
    splits, _ = build_splits(data_cfg(root))
    rows = {name: {tuple(x[0, :5].tolist()) for x in splits[name][0]} for name in splits}
    assert not rows["train"] & rows["val"] and not rows["train"] & rows["test"] and not rows["val"] & rows["test"]


def test_normalization_uses_training_folds_only(fake_ptbxl):
    root, (X, _, fold) = fake_ptbxl
    _, norm = build_splits(data_cfg(root))
    train = X[np.isin(fold, range(1, 9))]
    assert np.allclose(norm["mean"], train.mean(axis=(0, 1), keepdims=True))
    assert not np.allclose(norm["mean"], X.mean(axis=(0, 1), keepdims=True))  # would differ if test leaked in


def test_splits_are_channels_first_float_tensors(fake_ptbxl):
    root, _ = fake_ptbxl
    splits, _ = build_splits(data_cfg(root))
    x, y = splits["train"]
    assert x.dtype == torch.float32 and x.shape[1:] == (12, 1000)
    assert y.shape[1:] == (len(CLASSES),)


def test_normalize_gives_zero_mean_unit_std():
    X = np.random.default_rng(0).normal(3.0, 2.0, (50, 1000, 12)).astype(np.float32)
    out = normalize(X, X.mean(axis=(0, 1), keepdims=True), X.std(axis=(0, 1), keepdims=True))
    assert out.shape == (50, 12, 1000)
    assert torch.allclose(out.mean(dim=(0, 2)), torch.zeros(12), atol=1e-4)
    assert torch.allclose(out.std(dim=(0, 2)), torch.ones(12), atol=1e-3)


def test_auroc_perfect_and_inverted():
    y = np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 0], [1, 1, 0, 0, 1], [0, 0, 1, 1, 0]], dtype=float)
    macro, per_class = auroc(y, y)
    assert macro == 1.0 and set(per_class) == set(CLASSES)
    assert auroc(1 - y, y)[0] == 0.0


def test_macro_is_the_mean_of_classes():
    rng = np.random.default_rng(0)
    y = (rng.random((200, 5)) < 0.5).astype(float)
    macro, per_class = auroc(rng.random((200, 5)), y)
    assert macro == pytest.approx(np.mean(list(per_class.values())))


def test_config_overrides_parse_types(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump({"train": {"epochs": 30, "lr": 0.003}, "model": {"widths": [64]},
                                    "probe": {"enabled": True}}))
    cfg = load_config(path, ["train.epochs=5", "train.lr=1e-4", "model.widths=[8,8]", "probe.enabled=false"])
    assert cfg["train"]["epochs"] == 5 and cfg["train"]["lr"] == pytest.approx(1e-4)
    assert cfg["model"]["widths"] == [8, 8] and cfg["probe"]["enabled"] is False


@pytest.mark.parametrize("name", ["default", "shuffle_trained", "leaky_relu", "tanh"])
def test_shipped_configs_differ_from_default_only_where_intended(name):
    base = load_config("configs/default.yaml")
    cfg = load_config(f"configs/{name}.yaml")
    expected = {"default": {}, "shuffle_trained": {"augment.shuffle": 10},
                "leaky_relu": {"model.activation": "leaky_relu", "model.negative_slope": 0.01},
                "tanh": {"model.activation": "tanh"}}[name]

    def flat(d, prefix=""):
        return {k2: v2 for k, v in d.items()
                for k2, v2 in (flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v}).items()}

    a, b = flat(base), flat(cfg)
    diff = {k: b.get(k) for k in a.keys() | b.keys() if a.get(k) != b.get(k) and k != "name"}
    assert diff == expected
