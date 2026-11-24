import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import yaml

from crowdloc.irls import irls_solve
from crowdloc.init import init_weighted_centroid, init_trimmed_centroid
from crowdloc.metrics import cep90
from crowdloc.robust_kernels import make_prior_weights
from crowdloc.scene_gate import detect_scene
from crowdloc.synth import make_static_case, make_moving_case


def build_validation_set(seed=7, n_static=20, n_moving=20):
    rng = np.random.default_rng(seed)
    cases = [make_static_case(rng) for _ in range(n_static)]
    cases += [make_moving_case(rng) for _ in range(n_moving)]
    return cases


def evaluate_config(config, cases):
    errors_static, errors_moving = [], []
    for case in cases:
        anchors = case["anchors"]
        d = case["distances"]
        prior = case["prior_weights"] if "prior_weights" in case else np.ones(len(anchors))
        if isinstance(prior, np.ndarray) and prior.ndim == 1 and len(prior) == len(anchors):
            w = prior
        else:
            w = make_prior_weights(case.get("meta", {"time_decay": np.ones(len(anchors))}))

        scene = detect_scene(anchors, d, w, config["scene_gate"]["residual_mad_ratio_threshold"])
        if config.get("force_scene") and scene != config["force_scene"]:
            continue

        if config["init"] == "weighted_centroid":
            x0 = init_weighted_centroid(anchors, w)
        else:
            x0 = init_trimmed_centroid(anchors, w)

        # 移动场景：用最后一批时间戳的点作为初始锚点子集（简化处理）
        if scene == "moving":
            n_recent = max(5, len(anchors) // 3)
            anchors_use, d_use, w_use = anchors[-n_recent:], d[-n_recent:], w[-n_recent:]
        else:
            anchors_use, d_use, w_use = anchors, d, w

        res = irls_solve(anchors_use, d_use, w_use, x0, kernel=config["kernel"],
                         max_iter=config["meta"]["max_irls_iter"],
                         tol=config["meta"]["tol"],
                         scale_multiplier=config["scale_multiplier"])
        err = np.linalg.norm(res["x"] - case["true_x"])
        if case["scene"] == "static":
            errors_static.append(err)
        else:
            errors_moving.append(err)

    return {
        "static_cep90": cep90(np.array(errors_static)) if errors_static else float("inf"),
        "moving_cep90": cep90(np.array(errors_moving)) if errors_moving else float("inf"),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.example.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    cases = build_validation_set()
    base_static = evaluate_config({**cfg, "kernel": "huber", "init": "weighted_centroid",
                                   "scale_multiplier": 2.0, "force_scene": "static"}, cases)

    results = []
    for kernel, init, sm in itertools.product(
        cfg["meta"]["kernels"],
        ["weighted_centroid", "trimmed_centroid"],
        cfg["meta"]["scale_multipliers"],
    ):
        trial_cfg = {
            **cfg,
            "kernel": kernel,
            "init": init,
            "scale_multiplier": sm,
            "force_scene": None,
        }
        metrics = evaluate_config(trial_cfg, cases)
        gain = (base_static["static_cep90"] - metrics["moving_cep90"]) / max(base_static["static_cep90"], 1e-6)
        results.append({"config": {"kernel": kernel, "init": init, "scale_multiplier": sm},
                        "metrics": metrics, "moving_gain_vs_static": gain})

    results.sort(key=lambda r: (r["metrics"]["static_cep90"] + 0.5 * r["metrics"]["moving_cep90"]))
    best = results[0]
    Path(cfg["output"]["dir"]).mkdir(parents=True, exist_ok=True)
    with open(Path(cfg["output"]["dir"]) / "meta_iter_result.json", "w") as f:
        json.dump({"best": best, "all": results}, f, indent=2, ensure_ascii=False)
    print(json.dumps(best, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
