import numpy as np
from crowdloc.robust_kernels import kernel_weight, mad_scale


def weighted_least_squares(x0: np.ndarray, anchors: np.ndarray,
                           distances: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """闭式加权最小二乘，给定权重。用于 IRLS 的每一步子问题。"""
    diff = x0[None, :] - anchors
    r = np.linalg.norm(diff, axis=1)
    r = np.maximum(r, 1e-9)
    A = (anchors - x0[None, :]) / r[:, None]
    b = distances - r + A @ x0
    Aw = A * weights[:, None]
    return np.linalg.lstsq(Aw, b * weights, rcond=None)[0]


def irls_solve(anchors: np.ndarray, distances: np.ndarray,
               prior_weights: np.ndarray, x_init: np.ndarray,
               kernel: str, max_iter: int = 50, tol: float = 1e-4,
               scale_multiplier: float = 2.0) -> dict:
    """
    内层自迭代：IRLS。
    返回最优位置、每轮权重、残差历史。
    """
    x = x_init.copy()
    w = prior_weights.copy()
    history = []
    for it in range(max_iter):
        w_prev = w.copy()
        x = weighted_least_squares(x, anchors, distances, w)
        r = distances - np.linalg.norm(x[None, :] - anchors, axis=1)
        scale = mad_scale(r) * scale_multiplier
        robust_w = kernel_weight(r, scale, kernel)
        w = prior_weights * robust_w
        w = np.clip(w, 1e-9, None)
        history.append({"iter": it, "x": x.copy(), "residuals": r.copy(),
                        "scale": scale, "weight_change": np.linalg.norm(w - w_prev)})
        if np.linalg.norm(w - w_prev) < tol:
            break
    return {"x": x, "weights": w, "history": history, "iterations": len(history)}
