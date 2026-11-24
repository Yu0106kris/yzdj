import numpy as np


def cep90(errors: np.ndarray) -> float:
    return float(np.percentile(errors, 90))


def moving_gain(static_cep90: float, moving_cep90: float) -> float:
    """移动场景相对静止基线的检测精度增益。"""
    if static_cep90 <= 0:
        return 0.0
    return (static_cep90 - moving_cep90) / static_cep90
