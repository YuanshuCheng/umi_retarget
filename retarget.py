"""Retarget pipeline: UMI 位姿 → 全身关节轨迹。"""
import glob
import json
import os
import time
from types import SimpleNamespace

import numpy as np

from .core.data_loader import (
    load_raw_episode, resample_uniform, apply_body_mapping, smooth_trajectory,
)
from .core.robot_loader import load_pyroki_robot
from .core.optimizer import solve_trajectory_pyroki
from .core.postprocess import (
    build_output, validate_physics, trim_static,
    gen_action_chunks, compute_quality, compute_normalization, save_dataset,
)


def _make_cfg(mode, hand, urdf_path, weights):
    cfg = dict(weights)
    cfg.update(mode=mode, hand=hand, urdf_path=urdf_path, output_dir="")
    return SimpleNamespace(**cfg)


def preprocess_episode(path, cfg):
    ep = load_raw_episode(path)
    if ep is None:
        return None
    if len(ep["timestamps"]) < int(getattr(cfg, "min_episode_duration", 2.0) * 30):
        return None
    ep = resample_uniform(ep, getattr(cfg, "target_freq", 30))
    if ep is None:
        return None
    ep = apply_body_mapping(ep, cfg)
    ep = smooth_trajectory(ep, cfg)
    return ep


def gpu_solve_and_save(ep, robot_info, cfg, output_dir):
    ep = solve_trajectory_pyroki(ep, robot_info, cfg)
    if "optimized_joints" not in ep:
        return None
    ep = build_output(ep, cfg)
    ep = validate_physics(ep, cfg)
    ep = trim_static(ep, threshold=getattr(cfg, "trim_static_threshold", 0.001))
    ep = gen_action_chunks(ep, getattr(cfg, "action_chunk_size", 50))
    _, details = compute_quality(ep, cfg)
    ep["quality"] = details
    details["index"] = 0
    details["source"] = ep.get("source", "")
    details["status"] = "PASS" if details.get("final_score", 0) >= 0.7 else "WARN"
    norm = compute_normalization([ep])
    cfg_dict = vars(cfg).copy()
    cfg_dict["output_dir"] = output_dir
    cfg_save = SimpleNamespace(**cfg_dict)
    save_dataset([ep], norm, [details], cfg_save)
    tracking = ep.get("tracking", {})
    return {"details": details, "tracking": tracking}


def retarget_single(input_path, output_dir, cfg, robot_info):
    ep = preprocess_episode(input_path, cfg)
    if ep is None:
        return None
    os.makedirs(output_dir, exist_ok=True)
    return gpu_solve_and_save(ep, robot_info, cfg, output_dir)


def retarget_batch(input_dir, output_dir, config, parallel=0, force=False):
    urdf_path = config["urdf_path"]
    weights = config.get("weights", {})
    subsets = config.get("subsets", {})

    print("加载 URDF...")
    robot_info = load_pyroki_robot(urdf_path)

    episodes = []
    for subset_name, params in subsets.items():
        sub_dir = os.path.join(input_dir, subset_name)
        if not os.path.isdir(sub_dir):
            continue
        mode = params.get("mode", "mobile")
        hand = params.get("hand", "left")
        cfg = _make_cfg(mode, hand, urdf_path, weights)

        for path in sorted(glob.glob(os.path.join(sub_dir, "episode_*.hdf5"))):
            ep_name = os.path.splitext(os.path.basename(path))[0]
            out = os.path.join(output_dir, subset_name, ep_name)
            if not force and os.path.exists(os.path.join(out, "dataset.hdf5")):
                print("  跳过 {}/{} (已处理)".format(subset_name, ep_name))
                continue
            episodes.append((path, subset_name, ep_name, cfg))

    print("待处理: {} episodes".format(len(episodes)))
    if not episodes:
        print("无需处理。")
        return []

    all_results = []
    done = 0
    total = len(episodes)
    t_start = time.monotonic()

    for path, subset_name, ep_name, cfg in episodes:
        done += 1
        out = os.path.join(output_dir, subset_name, ep_name)
        print("  [{}/{}] {}/{} ...".format(done, total, subset_name, ep_name))

        result = retarget_single(path, out, cfg, robot_info)
        if result:
            d = result.get("details", {})
            tracking = result.get("tracking", {})
            print("    → score={:.3f} [{}]".format(
                d.get("final_score", 0), d.get("status", "")))
            all_results.append({
                "subset": subset_name, "episode": ep_name, **tracking,
            })
        else:
            print("    → 失败")

    elapsed = time.monotonic() - t_start
    print("\n完成: {}/{}, 耗时 {:.1f}s ({:.1f}min)".format(
        len(all_results), total, elapsed, elapsed / 60))

    summary_path = os.path.join(output_dir, "batch_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("batch_summary → {}".format(summary_path))

    return all_results
