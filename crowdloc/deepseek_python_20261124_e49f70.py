import numpy as np


def detect_scene(anchors: np.ndarray, distances: np.ndarray,
                 prior_weights: np.ndarray, mad_ratio_threshold: float = 2.5) -> str:
    """
    极简场景门控：用先验加权残差的 MAD 与中位数绝对值的比值作为离散度信号。
    真实系统中可接入时间戳序列做运动一致性检验。
    """
    if len(anchors) < 5:
        return "static"
    # 用加权质心作为粗估计
    w = prior_weights / prior_weights.sum()
    x_c = (anchors * w[:, None]).sum(axis=0)
    r = distances - np.linalg.norm(x_c[None, :] - anchors, axis=1)
    med = np.median(np.abs(r))
    mad = np.median(np.abs(r - np.median(r)))
    ratio = mad / max(med, 1e-6)
    return "moving" if ratio > mad_ratio_threshold else "static"