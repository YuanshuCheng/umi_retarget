"""Dynamics 补偿 wrapper。"""
import os
import pickle

import numpy as np

from .core.compensation import predict_residual, compensate_trajectory
from .core.robot_loader import load_pyroki_robot
from .utils import find_datasets


def compensate_batch(input_dir, config):
    model_path = config.get("compensation", {}).get("model_path")
    if not model_path:
        print("未配置补偿模型, 跳过。")
        print("如需补偿, 在 config.yaml 中设置 compensation.model_path")
        return

    if not os.path.exists(model_path):
        print("补偿模型不存在: {}".format(model_path))
        return

    import h5py

    with open(model_path, "rb") as f:
        calib = pickle.load(f)
    print("模型已加载: {} ({}条log, {}帧)".format(
        model_path, calib.get("n_logs", "?"), calib.get("n_frames", "?")))

    urdf_path = config.get("urdf_path", "")
    robot_info = load_pyroki_robot(urdf_path)
    freq = calib.get("freq", 30.0)

    files = find_datasets(input_dir)
    print("待补偿: {} 个文件".format(len(files)))

    for fp in files:
        print("  处理: {}".format(fp))
        with h5py.File(fp, "r") as f:
            joint_pos = f["data/demo_0/obs/joint_positions"][:]
            actions = f["data/demo_0/actions"][:] if "data/demo_0/actions" in f else None

        q_cmd = joint_pos[:, :18]
        q_warped = compensate_trajectory(q_cmd, calib, robot_info, freq)

        joint_pos_new = joint_pos.copy()
        joint_pos_new[:, :18] = q_warped

        actions_new = None
        if actions is not None:
            actions_new = actions.copy()
            actions_new[:, :18] = q_warped

        with h5py.File(fp, "a") as f:
            if "data/demo_0/obs/joint_positions" in f:
                del f["data/demo_0/obs/joint_positions"]
            f.create_dataset("data/demo_0/obs/joint_positions", data=joint_pos_new)
            if actions_new is not None and "data/demo_0/actions" in f:
                del f["data/demo_0/actions"]
                f.create_dataset("data/demo_0/actions", data=actions_new)
            f.attrs["dynamics_compensation"] = model_path

        diff = np.abs(q_warped - q_cmd)
        print("    补偿量: mean={:.2f}° max={:.2f}°".format(
            np.degrees(diff.mean()), np.degrees(diff.max())))

    print("\n完成: {} 个文件".format(len(files)))
