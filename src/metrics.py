import numpy as np
from sklearn import metrics


def get_metrics(
    preds: list[int],
    labels: list[int],
    probs: np.array | None = None,
    multi_class: str = "ovr",
):
    metrics_dict = {}
    labels = np.asarray(labels)
    if probs is not None:
        auc = metrics.roc_auc_score(labels, probs, multi_class=multi_class)
        metrics_dict.update({"auc": auc})
    return metrics_dict
