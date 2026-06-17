"""config.yaml 读写 + 预设系统。"""
import glob
import os

import yaml
import h5py

PRESETS = {
    "balanced": {
        "target_freq": 30,
        "smooth_max_error": 0.01,
        "arm_scale": 1.0,
        "auto_calibrate": False,
        "num_iterations": 100,
        "pos_weight": 20,
        "ori_weight": 0.3,
        "kf_pos_weight": 200,
        "kf_ori_weight": 50,
        "smooth_weight": 5,
        "acc_weight": 5,
        "jerk_weight": 1,
        "limit_weight": 100,
        "collision_weight": 100,
        "backward_lean_weight": 100,
        "max_lean_deg": 50,
        "max_lean_weight": 20,
        "torso_tilt_weight": 10,
        "torso_yaw_weight": 1,
        "elbow_weight": 2,
        "rest_weight": 0.5,
        "wrist_rest_weight": 5,
        "base_pos_weight": 3,
        "base_yaw_weight": 0.5,
        "base_smooth_weight": 20,
        "torso_follow_weight": 100,
        "torso_follow_ratio": 1.0,
        "base_follow_arm_weight": 0.0,
        "action_chunk_size": 50,
        "quality_threshold": 0.5,
        "val_ratio": 0.1,
        "min_episode_duration": 2.0,
        "trim_static_threshold": 0.001,
        "swap_hands": False,
    },
}


def scan_subsets(input_dir):
    """扫描输入目录, 统计每个子集的 episode 数和帧数。"""
    result = {}
    for name in sorted(os.listdir(input_dir)):
        sub_dir = os.path.join(input_dir, name)
        if not os.path.isdir(sub_dir):
            continue
        eps = sorted(glob.glob(os.path.join(sub_dir, "*.hdf5")))
        if not eps:
            continue
        n_frames = 0
        for ep in eps:
            try:
                with h5py.File(ep, "r") as f:
                    n_frames += len(f["timestamps"])
            except Exception:
                pass
        result[name] = {"n_eps": len(eps), "n_frames": n_frames}
    return result


def make_default_config(urdf_path, subsets_config, preset="balanced"):
    """生成默认配置字典。"""
    return {
        "urdf_path": os.path.abspath(urdf_path),
        "preset": preset,
        "subsets": subsets_config,
        "weights": dict(PRESETS.get(preset, PRESETS["balanced"])),
        "compensation": {"model_path": None},
        "evaluation": {
            "fail_joint_limit_ratio": 0.1,
            "fail_collision_ratio": 0.05,
            "fail_stuck_seconds": 3.0,
            "fail_min_duration": 2.0,
            "fail_chassis_spin_deg": 360.0,
            "good_threshold": 0.7,
        },
    }


def load_config(path):
    """加载 config.yaml。"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config, path):
    """保存 config.yaml。"""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True,
                  sort_keys=False)
