"""后处理: build_output, validate_physics, trim, quality, normalization, save。"""
import os
import numpy as np
import h5py

from scipy.spatial import cKDTree

from .constants import (
    GRIPPER_JAW_OFFSET, GRIPPER_GRIP_MAX_FASTUMI, GRIPPER_GRIP_MAX_R1PRO,
    GRIP_TIGHTEN, JOINT_LIMITS, MAX_VEL, MAX_ACC,
    quat_to_rotmat, rotmat_to_quat,
)
from .optimizer import detect_keyframes

import jax.numpy as jnp


def build_output(ep, cfg):
    if "optimized_joints" not in ep:
        return ep

    optimized = ep["optimized_joints"]
    N = len(optimized)
    freq = ep["freq"]

    # 夹爪映射: FastUMI 开口(mm) → R1Pro 开口(mm) → 归一化[0,1]
    # FastUMI 最大开口 87mm (9cm), R1Pro 最大开口 80mm (8cm)
    # 收紧系数 0.9: 真机略微收紧，确保夹稳
    FASTUMI_GRIP_MAX = 87.0
    R1PRO_GRIP_MAX = 80.0
    GRIP_TIGHTEN = 0.95
    clamp_l = np.clip(ep["clamp_left"][:N], 0, FASTUMI_GRIP_MAX) * (R1PRO_GRIP_MAX / FASTUMI_GRIP_MAX) * GRIP_TIGHTEN / 100.0
    clamp_r = np.clip(ep["clamp_right"][:N], 0, FASTUMI_GRIP_MAX) * (R1PRO_GRIP_MAX / FASTUMI_GRIP_MAX) * GRIP_TIGHTEN / 100.0

    # 从优化结果提取各部分（需要根据 pyroki joint 顺序映射）
    # 暂时直接存 optimized_joints，具体映射在集成时调整
    joint_traj = np.zeros((N, 20))

    # 动态映射: 从 pyroki actuated_names 找到各关节索引
    act_names = ep.get("_actuated_names", [])
    def _find_act(name):
        for i, n in enumerate(act_names):
            if n == name: return i
        return -1

    for i in range(4):
        idx = _find_act("torso_joint{}".format(i+1))
        if idx >= 0: joint_traj[:, i] = optimized[:, idx]
    for i in range(7):
        idx = _find_act("left_arm_joint{}".format(i+1))
        if idx >= 0: joint_traj[:, 4+i] = optimized[:, idx]
    for i in range(7):
        idx = _find_act("right_arm_joint{}".format(i+1))
        if idx >= 0: joint_traj[:, 11+i] = optimized[:, idx]
    joint_traj[:, 18] = clamp_l
    joint_traj[:, 19] = clamp_r

    ep["joint_traj"] = joint_traj

    # 从优化结果提取底盘关节
    base_idxs = ep.get("_base_joint_indices", [])
    if len(base_idxs) == 3:
        base_traj = np.zeros((N, 3))
        for i, idx in enumerate(base_idxs):
            base_traj[:, i] = optimized[:, idx]
        ep["base_traj"] = base_traj
    else:
        ep["base_traj"] = np.zeros((N, 3))

    ep["init_pose"] = {
        "base": ep["base_traj"][0].copy(),
        "torso": joint_traj[0, 0:4].copy(),
        "arm_left": joint_traj[0, 4:11].copy(),
        "arm_right": joint_traj[0, 11:18].copy(),
    }

    br = ep["base_traj"].copy()
    br -= br[0]  # 相对首帧
    ep["base_traj_rel"] = br

    return ep


# ===================================================================
# Step 6: 物理验证（简化，pyroki 已保证大部分）
# ===================================================================
def validate_physics(ep, cfg):
    jt = ep["joint_traj"]; N = jt.shape[0]; dt = ep["dt"]
    valid = np.ones(N, dtype=bool); joints = jt[:,:18]
    for j in range(18):
        lo, hi = JOINT_LIMITS[j]
        valid[(joints[:,j]<lo-0.01)|(joints[:,j]>hi+0.01)] = False
    if N > 1:
        vel = np.abs(np.diff(joints, axis=0)/dt)
        for t in range(N-1):
            if np.any(vel[t] > MAX_VEL * 1.5): valid[t+1] = False
    ep["valid_mask"] = valid
    ni = int(np.sum(~valid))
    print("    物理验证: {}/{} 通过 ({:.1%})".format(N-ni, N, 1-ni/N if N else 0))
    return ep


def trim_static(ep, threshold=0.001, min_remain=30):
    jt = ep["joint_traj"]; N = jt.shape[0]
    if N < min_remain: return ep
    diffs = np.linalg.norm(np.diff(jt[:,:18], axis=0), axis=1)
    start = 0; end = N
    for i in range(len(diffs)):
        if diffs[i] > threshold: start = i; break
    for i in range(len(diffs)-1, -1, -1):
        if diffs[i] > threshold: end = i+2; break
    if end-start < min_remain: return ep
    for k in list(ep.keys()):
        v = ep[k]
        if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == N:
            ep[k] = v[start:end].copy()
    return ep


def compute_quality(ep, cfg):
    jt = ep["joint_traj"]; N = jt.shape[0]; dt = ep["dt"]
    d = {}
    d["valid_rate"] = float(np.mean(ep.get("valid_mask", np.ones(N, dtype=bool))))
    d["duration_sec"] = float(N*dt)
    d["motion_range"] = float(np.mean(np.ptp(jt[:,:18], axis=0)))
    tk = ep.get("tracking", {})
    d["tracking_pass_rate"] = tk.get("pass_rate", 1.0)
    d["tracking_max_err"] = tk.get("max_error", 0.0)
    d["tracking_mean_err"] = tk.get("mean_error", 0.0)
    score = (0.25*d["valid_rate"] + 0.25*d["tracking_pass_rate"]
             + 0.20*max(0, 1-d["tracking_max_err"]/0.01)
             + 0.15*min(1, d["motion_range"]/0.5)
             + 0.15*min(1, d["duration_sec"]/5.0))
    d["final_score"] = float(score)
    return score, d


# ===================================================================
# Step 7: 输出（复用 v1）
# ===================================================================
def gen_action_chunks(ep, K):
    a = ep["joint_traj"]; N = a.shape[0]
    ch = np.zeros((N,K,20))
    for t in range(N):
        for k in range(K): ch[t,k] = a[min(t+1+k, N-1)]
    ep["action_chunks"] = ch; return ep


def compute_normalization(episodes):
    st = {}
    acts = np.concatenate([e["joint_traj"] for e in episodes], axis=0)
    st["action_min"]=acts.min(0).tolist(); st["action_max"]=acts.max(0).tolist()
    st["action_mean"]=acts.mean(0).tolist(); st["action_std"]=acts.std(0).tolist()
    bases = [e["base_traj_rel"] for e in episodes if "base_traj_rel" in e]
    if bases:
        b = np.concatenate(bases, axis=0)
        st["base_min"]=b.min(0).tolist(); st["base_max"]=b.max(0).tolist()
        st["base_mean"]=b.mean(0).tolist(); st["base_std"]=b.std(0).tolist()
    return st


def save_dataset(episodes, norm, qr, cfg):
    od = Path(cfg.output_dir); od.mkdir(parents=True, exist_ok=True)
    hp = od/"dataset.hdf5"; ne = len(episodes)
    idx = list(range(ne)); np.random.seed(42); np.random.shuffle(idx)
    nv = max(1, int(ne*cfg.val_ratio))
    vi = sorted(idx[:nv]); ti = sorted(idx[nv:])
    print("\n保存 → {} ({} eps)".format(hp, ne))
    with h5py.File(str(hp), "w") as f:
        f.attrs["total_episodes"]=ne; f.attrs["freq_hz"]=cfg.target_freq
        f.attrs["action_dim"]=20; f.attrs["base_dim"]=3
        f.attrs["layout"]="torso(4)+left_arm(7)+right_arm(7)+grip_l(1)+grip_r(1)"
        f.attrs["mode"]=cfg.mode; f.attrs["solver"]="pyroki"
        data = f.create_group("data")
        for i, ep in enumerate(episodes):
            d = data.create_group("demo_{}".format(i))
            N = ep["joint_traj"].shape[0]
            d.attrs["success"]=ep.get("success",True); d.attrs["task"]=ep.get("task","")
            d.attrs["num_steps"]=N; d.attrs["source_file"]=ep.get("source","")
            tk=ep.get("tracking",{}); d.attrs["tracking_max_error"]=tk.get("max_error",0)
            r = next((x for x in qr if x.get("index")==i), {})
            d.attrs["quality_score"]=r.get("final_score",0)
            if "init_pose" in ep:
                ip = d.create_group("init_pose")
                for k,v in ep["init_pose"].items():
                    ip.create_dataset(k, data=np.array(v, dtype=np.float64))
            obs = d.create_group("obs")
            obs.create_dataset("joint_positions", data=ep["joint_traj"])
            if "base_traj_rel" in ep:
                obs.create_dataset("base_position", data=ep["base_traj_rel"])
            for s in ["left","right"]:
                pk_name="pose_"+s
                if pk_name in ep: obs.create_dataset("eef_pose_"+s, data=ep[pk_name][:N])
            d.create_dataset("actions", data=ep["joint_traj"])
            if "base_traj_rel" in ep:
                d.create_dataset("base_actions", data=ep["base_traj_rel"])
            if "action_chunks" in ep:
                d.create_dataset("action_chunks", data=ep["action_chunks"])
            if "tracking_errors" in ep:
                qg = d.create_group("quality")
                te = ep["tracking_errors"]
                for k in ["left_frame", "right_frame", "left_overlap", "right_overlap"]:
                    if k in te: qg.create_dataset("tracking_" + k, data=te[k][:N])
                if "keyframe_mask" in te:
                    qg.create_dataset("keyframe_mask", data=te["keyframe_mask"][:N])
        ng = f.create_group("normalization")
        for k,v in norm.items(): ng.create_dataset(k, data=np.array(v, dtype=np.float64))
        mg = f.create_group("mask")
        mg.create_dataset("train", data=np.array(ti, dtype=np.int64))
        mg.create_dataset("valid", data=np.array(vi, dtype=np.int64))
    with open(str(od/"normalization.json"),"w") as nf: json.dump(norm, nf, indent=2)
    with open(str(od/"quality_report.json"),"w") as rf: json.dump(qr, rf, indent=2, ensure_ascii=False)
    print("完成。")


# ===================================================================
# 单 episode 处理
