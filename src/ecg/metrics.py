import numpy as np
from sklearn.metrics import roc_auc_score

from ecg import CLASSES


def auroc(probs, targets):
    """Return (macro AUROC, per-class AUROC dict)."""
    per_class = {c: float(roc_auc_score(targets[:, i], probs[:, i])) for i, c in enumerate(CLASSES)}
    return float(np.mean(list(per_class.values()))), per_class
