import numpy as np


def init_weighted_centroid(anchors, weights):
    w = weights / weights.sum()
    return (anchors * w[:, None]).sum(axis=0)


def init_trimmed_centroid(anchors, weights, trim_ratio=0.3):
    n = len(anchors)
    k = max(2, int(n * (1 - trim_ratio)))
    dists = np.linalg.norm(anchors - np.median(anchors, axis=0), axis=1)
    idx = np.argsort(dists)[:k]
    return np.average(anchors[idx], axis=0, weights=weights[idx])
