"""1D ResNet for 12-lead ECG classification."""
import torch
import torch.nn as nn

from ecg import CLASSES


def make_act(act="relu", negative_slope=0.01):
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "leaky_relu":
        return nn.LeakyReLU(negative_slope, inplace=True)
    if act == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {act}")


class ResidualBlock(nn.Module):
    def __init__(self, cin, cout, stride, k=7, act="relu", negative_slope=0.01):
        super().__init__()
        p = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, stride, p, bias=False),
            nn.BatchNorm1d(cout),
            make_act(act, negative_slope),
            nn.Conv1d(cout, cout, k, 1, p, bias=False),
            nn.BatchNorm1d(cout),
        )
        if stride == 1 and cin == cout:
            self.skip = nn.Identity()
        else:
            self.skip = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.BatchNorm1d(cout))
        self.act = make_act(act, negative_slope)

    def forward(self, x):
        return self.act(self.net(x) + self.skip(x))


class ResNet1D(nn.Module):
    def __init__(self, n_leads=12, n_classes=len(CLASSES), widths=(64, 128, 256, 256),
                 kernel_size=7, dropout=0.3, act="relu", negative_slope=0.01):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, widths[0], 15, 2, 7, bias=False),
            nn.BatchNorm1d(widths[0]),
            make_act(act, negative_slope),
            nn.MaxPool1d(3, 2, 1),
        )
        layers, cin = [], widths[0]
        for i, w in enumerate(widths):
            stride = 1 if i == 0 else 2
            layers += [ResidualBlock(cin, w, stride, kernel_size, act, negative_slope),
                       ResidualBlock(w, w, 1, kernel_size, act, negative_slope)]
            cin = w
        self.layers = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(cin * 2, n_classes))

    def forward(self, x):
        x = self.layers(self.stem(x))
        x = torch.cat([x.mean(-1), x.amax(-1)], dim=1)  # average + max pooling over time
        return self.head(x)


def build_model(cfg):
    # old configs have no activation keys, so they still build the original ReLU model
    return ResNet1D(widths=tuple(cfg["widths"]), kernel_size=cfg["kernel_size"], dropout=cfg["dropout"],
                    act=cfg.get("activation", "relu"), negative_slope=cfg.get("negative_slope", 0.01))
