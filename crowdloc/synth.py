import numpy as np


def make_static_case(rng, n_good=12, n_outlier=3, noise_std=8.0, outlier_bias=60.0):
    true_x = rng.uniform(-50, 50, size=2)
    good = true_x + rng.normal(0, 25, size=(n_good, 2))
    bad = true_x + rng.normal(0, 25, size=(n_outlier, 2)) + rng.normal(0, outlier_bias, size=(n_outlier, 2))
    anchors = np.vstack([good, bad])
    d = np.linalg.norm(anchors - true_x[None, :], axis=1) + rng.normal(0, noise_std, size=len(anchors))
    w = np.ones(len(anchors))
    return {"true_x": true_x, "anchors": anchors, "distances": d, "prior_weights": w, "scene": "static"}


def make_moving_case(rng, n_steps=6, n_per_step=4, speed=15.0):
    start = rng.uniform(-50, 50, size=2)
    direction = rng.normal(size=2)
    direction /= np.linalg.norm(direction)
    anchors, distances, weights, true_positions = [], [], [], []
    for t in range(n_steps):
        pos = start + direction * speed * t
        n = n_per_step + rng.integers(-1, 2)
        a = pos + rng.normal(0, 20, size=(n, 2))
        d = np.linalg.norm(a - pos[None, :], axis=1) + rng.normal(0, 8.0, size=n)
        anchors.append(a); distances.append(d)
        weights.append(np.ones(n)); true_positions.append(pos)
    return {
        "true_x": np.array(true_positions[-1]),  # 目标是最后时刻位置
        "anchors": np.vstack(anchors), "distances": np.concatenate(distances),
        "prior_weights": np.concatenate(weights), "scene": "moving",
    }
