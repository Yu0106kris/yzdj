import numpy as np


def mad_scale(residuals: np.ndarray) -> float:
    med = np.median(residuals)
    mad = np.median(np.abs(residuals - med))
    return max(1.4826 * mad, 1e-6)


def kernel_weight(r: np.ndarray, scale: float, kind: str) -> np.ndarray:
    """返回 IRLS 等价权，r 为残差向量。"""
    u = np.abs(r) / max(scale, 1e-9)
    if kind == "huber":
        c = 1.345
        return np.where(u <= c, 1.0, c / np.maximum(u, 1e-9))
    if kind == "cauchy":
        c = 2.3849
        return 1.0 / (1.0 + (u / c) ** 2)
    if kind == "tukey":
        c = 4.685
        return np.where(u <= c, (1.0 - (u / c) ** 2) ** 2, 0.0)
    if kind == "geman_mcclure":
        c = 2.0
        return 1.0 / (1.0 + (u / c) ** 2) ** 2
    raise ValueError(f"Unknown kernel: {kind}")


def make_prior_weights(meta: dict) -> np.ndarray:
    """
    将赛题中的 ω_i 先验信息合成为初始权重。
    预期 meta 包含 time_decay、accuracy_score、confidence 等字段。
    这里给出一个可替换的默认合成方式。
    """
    n = len(meta["time_decay"])
    w = np.ones(n, dtype=float)
    for key in ("time_decay", "accuracy_score", "confidence"):
        if key in meta:
            w *= np.asarray(meta[key], dtype=float)
    w = np.clip(w, 1e-3, None)
    return w / w.mean()
