"""数据加载、重采样、坐标变换、平滑。"""
import glob
import os

import numpy as np
import h5py
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation, Slerp

from .constants import (
    FASTUMI_TO_WORLD, FASTUMI_CAM_OFFSET,
    quat_to_rotmat, rotmat_to_quat, fastumi_pose_to_world,
)


def discover_episodes(input_dir):
    paths = sorted(glob.glob(os.path.join(input_dir, "**/episode.hdf5"), recursive=True))
    if not paths:
        paths = sorted(glob.glob(os.path.join(input_dir, "**/*.hdf5"), recursive=True))
    return paths


def load_raw_episode(path):
    try:
        with h5py.File(str(path), "r") as f:
            if "raw_input" not in f:
                print("  跳过 {} — 无 raw_input".format(path))
                return None
            ep = {
                "timestamps": f["timestamps"][:],
                "pose_left": f["raw_input/fastumi_pose_left"][:],
                "pose_right": f["raw_input/fastumi_pose_right"][:],
                "clamp_left": f["raw_input/clamp_left"][:],
                "clamp_right": f["raw_input/clamp_right"][:],
                "success": bool(f.attrs.get("success", True)),
                "task": str(f.attrs.get("task", "unspecified")),
                "source": str(path),
            }
            for cam in ["wrist_left_rgb", "wrist_right_rgb"]:
                ep["has_" + cam] = "observations/images/" + cam in f
            return ep
    except Exception as e:
        print("  跳过 {} — {}".format(path, e))
        return None


def resample_uniform(ep, freq):
    ts = ep["timestamps"]
    if len(ts) < 2:
        return None
    ts = ts - ts[0]
    dur = ts[-1]
    if dur < 0.1:
        return None
    N = max(2, int(round(dur * freq)) + 1)
    ts_new = np.linspace(0, dur, N)
    out = {"timestamps": ts_new, "dt": 1.0 / freq, "freq": freq}
    for side in ["left", "right"]:
        p = ep["pose_" + side]
        quat_norms = np.linalg.norm(p[:, 3:7], axis=1)
        if np.all(quat_norms < 1e-6):
            out["pose_" + side] = np.zeros((N, 7))
            out["pose_" + side][:, 6] = 1.0
            continue
        pos = np.column_stack([np.interp(ts_new, ts, p[:, i]) for i in range(3)])
        quat = Slerp(ts, Rotation.from_quat(p[:, 3:7]))(ts_new).as_quat()
        out["pose_" + side] = np.hstack([pos, quat])
    for side in ["left", "right"]:
        out["clamp_" + side] = np.interp(ts_new, ts, ep["clamp_" + side])
    for k in ["success", "task", "source", "has_wrist_left_rgb", "has_wrist_right_rgb"]:
        if k in ep:
            out[k] = ep[k]
    return out


def _compute_yaw_calibration(poses, freq=30):
    N = len(poses)
    window = min(N, int(2.0 * freq))
    if window < 10:
        return np.eye(3), 0.0
    delta = poses[window // 2:window, :3] - poses[:window // 2, :3]
    motion_xz = delta.mean(axis=0)[[0, 2]]
    mag = np.sqrt(motion_xz[0] ** 2 + motion_xz[1] ** 2)
    if mag < 0.02:
        return np.eye(3), 0.0
    yaw_offset = np.arctan2(motion_xz[0], motion_xz[1])
    c, s = np.cos(-yaw_offset), np.sin(-yaw_offset)
    R_calib = np.array([[c, 0, -s], [0, 1, 0], [s, 0, c]], dtype=np.float64)
    return R_calib, np.degrees(yaw_offset)


def apply_body_mapping(ep, cfg):
    for side in ["left", "right"]:
        poses = ep["pose_" + side]
        if getattr(cfg, "auto_calibrate", False):
            R_calib, yaw_deg = _compute_yaw_calibration(poses, freq=ep.get("freq", 30))
            if abs(yaw_deg) > 1.0:
                print("    坐标校准 ({}): yaw偏移={:.1f}°".format(side, yaw_deg))
                for t in range(len(poses)):
                    poses[t, :3] = R_calib @ poses[t, :3]
                    R_t = quat_to_rotmat(poses[t, 3:7])
                    poses[t, 3:7] = rotmat_to_quat(R_calib @ R_t)
        for t in range(len(poses)):
            q = poses[t, 3:7]
            if np.linalg.norm(q) > 1e-6:
                R_dev = quat_to_rotmat(q)
                poses[t, :3] += R_dev @ FASTUMI_CAM_OFFSET
        mapped = np.zeros_like(poses)
        for t in range(len(poses)):
            p, q = fastumi_pose_to_world(poses[t, :3], poses[t, 3:7])
            mapped[t, :3] = p
            mapped[t, 3:7] = q
        if abs(cfg.arm_scale - 1.0) > 1e-4:
            center = mapped[0, :3].copy()
            mapped[:, :3] = center + (mapped[:, :3] - center) * cfg.arm_scale
        ep["pose_" + side] = mapped
    return ep


def smooth_trajectory(ep, cfg):
    freq = ep["freq"]
    for side in ["left", "right"]:
        poses = ep["pose_" + side]
        pos = poses[:, :3].copy()
        quat = poses[:, 3:7].copy()
        best = pos.copy()
        for cutoff in [6, 8, 10, 12, 15, 20]:
            if len(pos) < 13:
                break
            b, a = butter(2, cutoff, fs=freq, btype="low")
            s = filtfilt(b, a, pos, axis=0)
            if np.linalg.norm(s - pos, axis=1).max() <= cfg.smooth_max_error:
                best = s
                break
        rots = Rotation.from_quat(quat)
        qs = quat.copy()
        hw = 2
        for t in range(hw, len(quat) - hw):
            qs[t] = Slerp(np.arange(2 * hw + 1), rots[t - hw:t + hw + 1])(hw).as_quat()
        ep["pose_" + side] = np.hstack([best, qs])
        ep["smooth_err_" + side] = np.linalg.norm(best - pos, axis=1)
    return ep
