"""pyroki 全局轨迹优化: EE 追踪 + 平滑 + 碰撞 + 拟人化。"""
import time

import numpy as np

import jax
import jax.numpy as jnp
import jaxls
import jaxlie
import pyroki as pk

from scipy.signal import butter, filtfilt
from scipy.spatial import cKDTree

from .constants import (
    GRIPPER_JAW_OFFSET, GRIPPER_GRIP_MAX_FASTUMI, GRIPPER_GRIP_MAX_R1PRO,
    GRIP_TIGHTEN, quat_to_rotmat, rotmat_to_quat,
)


def detect_keyframes(ep, N, freq, hand):
    """用夹爪变化率检测关键帧。"""
    kf_mask = np.zeros(N, dtype=bool)
    pre_window = int(0.5 * freq)
    post_window = int(0.3 * freq)
    skip_start = int(0.5 * freq)
    skip_end = int(0.5 * freq)
    change_thresh = 0.5
    for side in (["right"] if hand != "left" else []) + (["left"] if hand != "right" else []):
        clamp = ep["clamp_" + side][:N]
        dcl = np.abs(np.diff(clamp))
        change_frames = np.where(dcl > change_thresh)[0]
        for cf in change_frames:
            if cf < skip_start or cf > N - skip_end:
                continue
            s = max(0, cf - pre_window)
            e = min(N, cf + post_window)
            kf_mask[s:e] = True
    return kf_mask


def compute_world_targets_np(mapped_poses, fk_ref, init_offset=None, freq=30):
    """世界系 IK 目标。返回夹爪中心目标和法兰目标。"""
    N = len(mapped_poses)
    jaw_targets = np.zeros((N, 7))
    flange_targets = np.zeros((N, 7))
    pos0 = mapped_poses[0, :3]
    R_FK = quat_to_rotmat(fk_ref[3:7])
    R0_inv = quat_to_rotmat(mapped_poses[0, 3:7]).T
    offset_local = np.array([0.0, 0.0, GRIPPER_JAW_OFFSET])
    pos_offset = np.zeros(3) if init_offset is None else np.asarray(init_offset)
    ramp_frames = int(1.0 * freq)
    for t in range(N):
        ramp = min(1.0, t / max(1, ramp_frames))
        jaw_targets[t, :3] = fk_ref[:3] + (mapped_poses[t, :3] - pos0) + pos_offset * ramp
        R_cur = quat_to_rotmat(mapped_poses[t, 3:7])
        R_target = R_cur @ R0_inv @ R_FK
        jaw_targets[t, 3:7] = rotmat_to_quat(R_target)
        flange_targets[t, :3] = jaw_targets[t, :3] - R_target @ offset_local
        flange_targets[t, 3:7] = jaw_targets[t, 3:7]
    return jaw_targets, flange_targets


def get_fk_reference(robot_info):
    """获取 FK 初始位姿, 补偿到夹爪中心。"""
    robot = robot_info["robot"]
    n = robot_info["n_actuated"]
    q0 = np.zeros(n)
    if robot_info["left_elbow_idx"] >= 0:
        q0[robot_info["left_elbow_idx"]] = -0.15
    if robot_info["right_elbow_idx"] >= 0:
        q0[robot_info["right_elbow_idx"]] = -0.15
    fk = robot.forward_kinematics(jnp.array(q0))
    fk_arr = np.array(fk)

    def _flange_to_jaw(pos, wxyz):
        qx, qy, qz, qw = wxyz[1], wxyz[2], wxyz[3], wxyz[0]
        R = quat_to_rotmat(np.array([qx, qy, qz, qw]))
        return pos + R @ np.array([0.0, 0.0, GRIPPER_JAW_OFFSET])

    left_idx = robot_info["left_ee_idx"]
    right_idx = robot_info["right_ee_idx"]
    left_pos = fk_arr[left_idx, 4:7]
    left_wxyz = fk_arr[left_idx, 0:4]
    left_jaw = _flange_to_jaw(left_pos, left_wxyz)
    right_pos = fk_arr[right_idx, 4:7]
    right_wxyz = fk_arr[right_idx, 0:4]
    right_jaw = _flange_to_jaw(right_pos, right_wxyz)
    ref_l = np.concatenate([left_jaw, [left_wxyz[1], left_wxyz[2], left_wxyz[3], left_wxyz[0]]])
    ref_r = np.concatenate([right_jaw, [right_wxyz[1], right_wxyz[2], right_wxyz[3], right_wxyz[0]]])
    return ref_l, ref_r, q0


def solve_trajectory_pyroki(ep, robot_info, cfg):
    """pyroki 全局轨迹优化（基于 pyroki 官方 trajopt 模式）"""
    if not True:
        print("  pyroki 不可用，跳过优化")
        return ep

    import jax
    import jaxlie

    robot = robot_info["robot"]
    freq = ep["freq"]
    dt = 1.0 / freq
    N = len(ep["pose_left"])

    ref_l, ref_r, q0_np = get_fk_reference(robot_info)
    jaw_left, world_left = compute_world_targets_np(ep["pose_left"], ref_l, freq=freq)
    jaw_right, world_right = compute_world_targets_np(ep["pose_right"], ref_r, freq=freq)

    print("    初始FK(夹爪中心): L={} R={}".format(ref_l[:3].round(3), ref_r[:3].round(3)))
    print("    世界系EE范围: R x=[{:.3f},{:.3f}] z=[{:.3f},{:.3f}]".format(
        jaw_right[:,0].min(), jaw_right[:,0].max(),
        jaw_right[:,2].min(), jaw_right[:,2].max()))

    hand = cfg.hand
    left_ee_idx = robot_info["left_ee_idx"]
    right_ee_idx = robot_info["right_ee_idx"]
    n_act = robot_info["n_actuated"]

    # 初始配置（warm-start: 底盘yaw朝向目标 + 躯干yaw辅助）
    init_cfg = jnp.zeros((N, n_act))
    if robot_info["left_elbow_idx"] >= 0:
        init_cfg = init_cfg.at[:, robot_info["left_elbow_idx"]].set(-0.15)
    if robot_info["right_elbow_idx"] >= 0:
        init_cfg = init_cfg.at[:, robot_info["right_elbow_idx"]].set(-0.15)

    base_idxs_init = robot_info["base_joint_indices"]
    torso_all_init = [i for i in robot_info["torso_joint_indices"] if i >= 0]
    if cfg.mode == "mobile" and len(base_idxs_init) >= 3:
        ee_target = jaw_right if hand != "left" else jaw_left
        delta_xy = ee_target[:, :2] - ee_target[0, :2]
        dist_xy = np.sqrt(delta_xy[:, 0]**2 + delta_xy[:, 1]**2)

        # 初始化 base x/y: 跟随目标位移的 50%（保守，让优化器调整剩余部分）
        init_base_x = delta_xy[:, 0] * 0.5
        init_base_y = delta_xy[:, 1] * 0.5
        if N > 13:
            b_xy, a_xy = butter(2, 1, fs=freq, btype='low')
            init_base_x = filtfilt(b_xy, a_xy, init_base_x)
            init_base_y = filtfilt(b_xy, a_xy, init_base_y)
        x_idx = base_idxs_init[0]
        y_idx = base_idxs_init[1]
        init_cfg = init_cfg.at[:, x_idx].set(jnp.array(init_base_x))
        init_cfg = init_cfg.at[:, y_idx].set(jnp.array(init_base_y))

        # 初始化 base yaw: 位移方向 + unwrap 防止 ±180° 跳变
        target_dir = np.arctan2(delta_xy[:, 1], delta_xy[:, 0])
        target_dir[dist_xy < 0.05] = 0.0
        target_dir = np.unwrap(target_dir)
        if N > 13:
            b_init, a_init = butter(2, 1, fs=freq, btype='low')
            target_dir = filtfilt(b_init, a_init, target_dir)
        yaw_idx = base_idxs_init[2]
        init_cfg = init_cfg.at[:, yaw_idx].set(jnp.array(target_dir))

        # 初始化 torso j4: yaw 的 30%
        if len(torso_all_init) > 3:
            j4_idx = torso_all_init[3]
            init_cfg = init_cfg.at[:, j4_idx].set(jnp.array(target_dir * 0.3))

        print("    warm-start: base_xy=[x:{:.3f}~{:.3f} y:{:.3f}~{:.3f}]".format(
            init_base_x.min(), init_base_x.max(), init_base_y.min(), init_base_y.max()))
        print("    warm-start: base_yaw=[{:.1f}°,{:.1f}°] torso_j4=[{:.1f}°,{:.1f}°]".format(
            np.degrees(target_dir.min()), np.degrees(target_dir.max()),
            np.degrees(target_dir.min()*0.3), np.degrees(target_dir.max()*0.3)))

    # 关键帧检测（夹爪变化率 ± 窗口）
    kf_mask = detect_keyframes(ep, N, freq, hand)
    kf_indices = np.where(kf_mask)[0]
    n_kf = len(kf_indices)
    print("    关键帧: {}/{} ({:.1%})".format(n_kf, N, n_kf / N))

    # 轨迹变量
    traj_vars = robot.joint_var_cls(jnp.arange(N))

    # robot 加 batch 维度（pyroki 要求）
    robot_batched = jax.tree.map(lambda x: x[None], robot)

    # --- 构建 costs ---
    costs = []

    # 1a. EE 追踪 cost — 全帧基础追踪
    if hand != "left":
        target_wxyz_r = jnp.array(world_right[:, [6,3,4,5]])
        target_pos_r = jnp.array(world_right[:, :3])
        target_se3_r = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz_r), target_pos_r)
        costs.append(pk.costs.pose_cost(
            robot_batched, traj_vars, target_se3_r,
            jnp.array(right_ee_idx)[None],
            jnp.array([cfg.pos_weight] * 3)[None],
            jnp.array([cfg.ori_weight] * 3)[None]))
    if hand != "right":
        target_wxyz_l = jnp.array(world_left[:, [6,3,4,5]])
        target_pos_l = jnp.array(world_left[:, :3])
        target_se3_l = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(target_wxyz_l), target_pos_l)
        costs.append(pk.costs.pose_cost(
            robot_batched, traj_vars, target_se3_l,
            jnp.array(left_ee_idx)[None],
            jnp.array([cfg.pos_weight] * 3)[None],
            jnp.array([cfg.ori_weight] * 3)[None]))

    # 1b. EE 追踪 cost — 关键帧高精度叠加
    if cfg.kf_pos_weight > 0 and n_kf > 2:
        kf_vars = robot.joint_var_cls(jnp.array(kf_indices))
        robot_kf_batched = jax.tree.map(lambda x: x[None], robot)
        if hand != "left":
            kf_se3_r = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_r[kf_indices]),
                target_pos_r[kf_indices])
            costs.append(pk.costs.pose_cost(
                robot_kf_batched, kf_vars, kf_se3_r,
                jnp.array(right_ee_idx)[None],
                jnp.array([cfg.kf_pos_weight] * 3)[None],
                jnp.array([cfg.kf_ori_weight] * 3)[None]))
        if hand != "right":
            kf_se3_l = jaxlie.SE3.from_rotation_and_translation(
                jaxlie.SO3(target_wxyz_l[kf_indices]),
                target_pos_l[kf_indices])
            costs.append(pk.costs.pose_cost(
                robot_kf_batched, kf_vars, kf_se3_l,
                jnp.array(left_ee_idx)[None],
                jnp.array([cfg.kf_pos_weight] * 3)[None],
                jnp.array([cfg.kf_ori_weight] * 3)[None]))
        print("    关键帧 cost: pos={} ori={} (叠加到全帧 pos={} ori={})".format(
            cfg.kf_pos_weight, cfg.kf_ori_weight, cfg.pos_weight, cfg.ori_weight))

    # 2. 关节限位 (软约束)
    costs.append(pk.costs.limit_cost(robot_batched, traj_vars,
                                     weight=jnp.array([cfg.limit_weight])[None]))

    # 2.5 自碰撞 cost
    robot_coll = robot_info.get("robot_coll")
    if robot_coll is not None and cfg.collision_weight > 0:
        def _safe_batch_coll(rc):
            """batch robot_coll, 保持 int 字段不变"""
            import jax_dataclasses as jdc
            with jdc.copy_and_mutate(rc, validate=False) as rc_b:
                rc_b.coll = jax.tree.map(lambda x: x[None], rc.coll)
                rc_b._geom_to_link_idx = rc._geom_to_link_idx[None]
            return rc_b
        robot_coll_b = _safe_batch_coll(robot_coll)
        costs.append(pk.costs.self_collision_cost(
            robot_batched, robot_coll_b, traj_vars,
            margin=0.03,
            weight=jnp.array([cfg.collision_weight])[None]))
        print("    自碰撞 cost: {} 对, margin=0.03m weight={}".format(
            len(robot_coll.active_idx_i), cfg.collision_weight))

    # 3. 轨迹平滑
    if N > 1:
        costs.append(pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, N)),
            robot.joint_var_cls(jnp.arange(0, N - 1)),
            jnp.array([cfg.smooth_weight])[None]))
    if N > 4:
        costs.append(pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, N - 2)),
            robot.joint_var_cls(jnp.arange(4, N)),
            robot.joint_var_cls(jnp.arange(3, N - 1)),
            robot.joint_var_cls(jnp.arange(1, N - 3)),
            robot.joint_var_cls(jnp.arange(0, N - 4)),
            dt, jnp.array([cfg.acc_weight])[None]))

    # 4. 零位偏好 (rest cost)
    costs.append(pk.costs.rest_cost(
        traj_vars, traj_vars.default_factory()[None], jnp.array([cfg.rest_weight])[None]))

    # 5. 躯干（俯仰/侧倾 vs 偏航分离）+ 肘部弯曲
    torso_all = [i for i in robot_info["torso_joint_indices"] if i >= 0]
    torso_tilt_idxs = jnp.array(torso_all[:3])  # j1/j2/j3 俯仰侧倾
    torso_yaw_idxs = jnp.array(torso_all[3:4]) if len(torso_all) > 3 else jnp.array([], dtype=int)
    elbow_idxs = jnp.array([i for i in [robot_info["left_elbow_idx"],
                                         robot_info["right_elbow_idx"]] if i >= 0])

    @jaxls.Cost.factory(name="TorsoTilt")
    def torso_tilt_cost(vals, var):
        q = vals[var]
        return (q[..., torso_tilt_idxs] * cfg.torso_tilt_weight).flatten()

    @jaxls.Cost.factory(name="TorsoYaw")
    def torso_yaw_cost(vals, var):
        q = vals[var]
        return (q[..., torso_yaw_idxs] * cfg.torso_yaw_weight).flatten()

    @jaxls.Cost.factory(name="ElbowBend")
    def elbow_cost(vals, var):
        q = vals[var]
        return ((q[..., elbow_idxs] - (-0.15)) * cfg.elbow_weight).flatten()

    # 腕关节零位偏好 (抑制腕关节过度使用，鼓励肩/肘主导)
    act_names = robot_info["actuated_names"]
    def _find_act(name):
        for i, n in enumerate(act_names):
            if n == name: return i
        return -1
    left_wrist_idxs = [_find_act("left_arm_joint{}".format(i)) for i in [5, 6, 7]]
    right_wrist_idxs = [_find_act("right_arm_joint{}".format(i)) for i in [5, 6, 7]]
    wrist_idxs = jnp.array([i for i in left_wrist_idxs + right_wrist_idxs if i >= 0])

    if len(wrist_idxs) > 0 and cfg.wrist_rest_weight > 0:
        @jaxls.Cost.factory(name="WristRest")
        def wrist_rest_cost(vals, var):
            q = vals[var]
            return (q[..., wrist_idxs] * cfg.wrist_rest_weight).flatten()
        costs.append(wrist_rest_cost(traj_vars))

    # 后倾惩罚: j1+j2 < 0 时惩罚, 鼓励前倾
    torso_pitch_idxs = [i for i in torso_all[:2] if i >= 0]
    if len(torso_pitch_idxs) == 2 and cfg.backward_lean_weight > 0:
        _pitch_idx0 = torso_pitch_idxs[0]
        _pitch_idx1 = torso_pitch_idxs[1]
        _blw = cfg.backward_lean_weight

        @jaxls.Cost.factory(name="BackwardLean")
        def backward_lean_cost(vals, var):
            q = vals[var]
            total_pitch = q[..., _pitch_idx0] + q[..., _pitch_idx1]
            return (jnp.maximum(-total_pitch, 0.0) * _blw).flatten()

        costs.append(backward_lean_cost(traj_vars))
        print("    后倾惩罚: weight={}".format(cfg.backward_lean_weight))

    # 前倾上限: j1+j2+j3 总前倾超过阈值时惩罚, 引导折叠下蹲
    torso_pitch_all = [i for i in torso_all[:3] if i >= 0]
    if len(torso_pitch_all) == 3 and cfg.max_lean_weight > 0:
        _ml_idx0 = torso_pitch_all[0]
        _ml_idx1 = torso_pitch_all[1]
        _ml_idx2 = torso_pitch_all[2]
        _ml_limit = jnp.radians(cfg.max_lean_deg)
        _ml_w = cfg.max_lean_weight

        @jaxls.Cost.factory(name="MaxLean")
        def max_lean_cost(vals, var):
            q = vals[var]
            total = q[..., _ml_idx0] + q[..., _ml_idx1] + q[..., _ml_idx2]
            return (jnp.maximum(total - _ml_limit, 0.0) * _ml_w).flatten()

        costs.append(max_lean_cost(traj_vars))
        print("    前倾上限: {}° weight={}".format(cfg.max_lean_deg, cfg.max_lean_weight))

    costs.append(torso_tilt_cost(traj_vars))
    if len(torso_yaw_idxs) > 0:
        costs.append(torso_yaw_cost(traj_vars))
    costs.append(elbow_cost(traj_vars))

    # 6. 单手模式: 冻结非活跃臂
    if hand == "right":
        freeze_idxs = jnp.array(robot_info["left_arm_indices"])
        freeze_vals = init_cfg[0, freeze_idxs] if len(freeze_idxs) > 0 else jnp.zeros(0)
        @jaxls.Cost.factory(kind="constraint_eq_zero", name="FreezeLeftArm")
        def freeze_arm(vals, var):
            q = vals[var]
            return (q[..., freeze_idxs] - freeze_vals).flatten()
        costs.append(freeze_arm(traj_vars))
    elif hand == "left":
        freeze_idxs = jnp.array(robot_info["right_arm_indices"])
        freeze_vals = init_cfg[0, freeze_idxs] if len(freeze_idxs) > 0 else jnp.zeros(0)
        @jaxls.Cost.factory(kind="constraint_eq_zero", name="FreezeRightArm")
        def freeze_arm(vals, var):
            q = vals[var]
            return (q[..., freeze_idxs] - freeze_vals).flatten()
        costs.append(freeze_arm(traj_vars))

    # 7. 底盘 cost（平移 vs 旋转分离）
    base_all_idxs = robot_info["base_joint_indices"]
    if len(base_all_idxs) >= 3:
        base_xy_idxs = jnp.array(base_all_idxs[:2])   # chassis_x, chassis_y
        base_yaw_idxs = jnp.array(base_all_idxs[2:3]) # chassis_yaw
        base_all = jnp.array(base_all_idxs)

        if cfg.mode == "local":
            base_pos_w = 1000.0
            base_yaw_w = 1000.0
            base_smooth_w = 0.0
        else:
            base_pos_w = cfg.base_pos_weight
            base_yaw_w = cfg.base_yaw_weight
            base_smooth_w = cfg.base_smooth_weight

        @jaxls.Cost.factory(name="BasePosPenalty")
        def base_pos_penalty(vals, var):
            q = vals[var]
            return (q[..., base_xy_idxs] * base_pos_w).flatten()
        costs.append(base_pos_penalty(traj_vars))

        @jaxls.Cost.factory(name="BaseYawPenalty")
        def base_yaw_penalty(vals, var):
            q = vals[var]
            return (q[..., base_yaw_idxs] * base_yaw_w).flatten()
        costs.append(base_yaw_penalty(traj_vars))

        if base_smooth_w > 0 and N > 1:
            @jaxls.Cost.factory(name="BaseSmooth")
            def base_smooth(vals, var_curr, var_prev):
                return ((vals[var_curr][..., base_all] - vals[var_prev][..., base_all])
                        * base_smooth_w).flatten()
            costs.append(base_smooth(
                robot.joint_var_cls(jnp.arange(1, N)),
                robot.joint_var_cls(jnp.arange(0, N - 1))))

        print("    底盘: mode={} pos_w={} yaw_w={} smooth_w={}".format(
            cfg.mode, base_pos_w, base_yaw_w, base_smooth_w))

        # 8. 躯干跟随底盘方向 (仅 mobile 模式)
        if cfg.mode != "local" and cfg.torso_follow_weight > 0 and len(torso_yaw_idxs) > 0:
            _follow_ratio = cfg.torso_follow_ratio
            _follow_w = cfg.torso_follow_weight
            _base_yaw_idx = base_all_idxs[2]
            _torso_j4_idx = torso_all[3]

            @jaxls.Cost.factory(name="TorsoFollowBase")
            def torso_follow_base(vals, var):
                q = vals[var]
                base_yaw = q[..., _base_yaw_idx]
                torso_yaw = q[..., _torso_j4_idx]
                return ((torso_yaw - _follow_ratio * base_yaw) * _follow_w).flatten()
            costs.append(torso_follow_base(traj_vars))
            print("    躯干跟随底盘: ratio={} weight={}".format(_follow_ratio, _follow_w))

        # 9. 底盘跟随手臂朝向旋转 (仅 mobile 模式)
        if cfg.mode != "local" and cfg.base_follow_arm_weight > 0:
            arm_targets = jaw_left if hand != "right" else jaw_right
            arm_yaw_arr = np.zeros(N)
            R0_arm = quat_to_rotmat(arm_targets[0, 3:7])
            for t in range(N):
                Rt = quat_to_rotmat(arm_targets[t, 3:7])
                R_delta = Rt @ R0_arm.T
                arm_yaw_arr[t] = np.arctan2(R_delta[1, 0], R_delta[0, 0])
            if hand == "both":
                arm_yaw_r = np.zeros(N)
                R0_r = quat_to_rotmat(jaw_right[0, 3:7])
                for t in range(N):
                    Rt_r = quat_to_rotmat(jaw_right[t, 3:7])
                    R_d = Rt_r @ R0_r.T
                    arm_yaw_r[t] = np.arctan2(R_d[1, 0], R_d[0, 0])
                arm_yaw_arr = (arm_yaw_arr + arm_yaw_r) / 2.0
            if N > 13:
                b_ay, a_ay = butter(2, 2, fs=freq, btype='low')
                arm_yaw_arr = filtfilt(b_ay, a_ay, arm_yaw_arr)

            arm_yaw_jnp = jnp.array(arm_yaw_arr.astype(np.float32))
            _base_yaw_follow_idx = base_all_idxs[2]
            _bfa_w = cfg.base_follow_arm_weight

            @jaxls.Cost.factory(name="BaseFollowArmYaw")
            def base_follow_arm(vals, var):
                q = vals[var]
                base_yaw = q[..., _base_yaw_follow_idx]
                return ((base_yaw - arm_yaw_jnp) * _bfa_w).flatten()
            costs.append(base_follow_arm(traj_vars))
            print("    底盘跟随手臂朝向: yaw范围=[{:.1f}°,{:.1f}°] weight={}".format(
                np.degrees(arm_yaw_arr.min()), np.degrees(arm_yaw_arr.max()),
                _bfa_w))

    print("    pyroki 求解: {} 帧, {} costs...".format(N, len(costs)))

    # --- 求解 ---
    t_analyze = time.monotonic()
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars]).analyze()
    print("    analyze (JIT编译): {:.1f}s".format(time.monotonic() - t_analyze))

    t_solve = time.monotonic()
    solution = problem.solve(
        initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_cfg),)),
        termination=jaxls.TerminationConfig(max_iterations=cfg.num_iterations),
    )
    elapsed = time.monotonic() - t_solve
    print("    solve (优化): {:.1f}s".format(elapsed))

    optimized = np.array(solution[traj_vars])
    total_time = time.monotonic() - t_analyze
    print("    求解完成: {:.1f}s ({} 帧)".format(total_time, N))

    # --- 后处理: 平滑起始段 (替换前 blend_frames 帧) ---
    blend_frames = min(60, N // 4)
    if blend_frames > 2:
        zero_cfg = np.zeros(n_act)
        if robot_info["left_elbow_idx"] >= 0:
            zero_cfg[robot_info["left_elbow_idx"]] = -0.15
        if robot_info["right_elbow_idx"] >= 0:
            zero_cfg[robot_info["right_elbow_idx"]] = -0.15
        target_cfg = optimized[blend_frames].copy()
        for t in range(blend_frames):
            s = t / blend_frames
            alpha = 10*s**3 - 15*s**4 + 6*s**5
            optimized[t] = zero_cfg * (1 - alpha) + target_cfg * alpha
        print("    起始平滑: 前{}帧替换为零位→t={}的最小jerk插值".format(blend_frames, blend_frames))

    # --- FK 验证 (用夹爪中心位置) ---
    from scipy.spatial import cKDTree

    offset_local = np.array([0.0, 0.0, GRIPPER_JAW_OFFSET])
    jaw_pos_l = np.zeros((N, 3)); jaw_pos_r = np.zeros((N, 3))
    ori_err_l = np.zeros(N); ori_err_r = np.zeros(N)
    for t in range(N):
        fk_arr = np.array(robot.forward_kinematics(jnp.array(optimized[t])))
        for side, ee_idx, jaw_pos, ori_err, jaw_tgt in [
                ("left", left_ee_idx, jaw_pos_l, ori_err_l, jaw_left),
                ("right", right_ee_idx, jaw_pos_r, ori_err_r, jaw_right)]:
            flange_pos = fk_arr[ee_idx, 4:7]
            flange_wxyz = fk_arr[ee_idx, 0:4]
            quat_fk = np.array([flange_wxyz[1], flange_wxyz[2], flange_wxyz[3], flange_wxyz[0]])
            R_fk = quat_to_rotmat(quat_fk)
            jaw_pos[t] = flange_pos + R_fk @ offset_local
            quat_tgt = jaw_tgt[t, 3:7]
            if np.linalg.norm(quat_tgt) > 1e-6:
                R_tgt = quat_to_rotmat(quat_tgt)
                R_diff = R_fk @ R_tgt.T
                cos_angle = np.clip((np.trace(R_diff) - 1) / 2, -1, 1)
                ori_err[t] = np.degrees(np.arccos(cos_angle))

    # 1. 逐帧位置误差
    err_frame_l = np.linalg.norm(jaw_pos_l - jaw_left[:, :3], axis=1) if hand != "right" else np.zeros(N)
    err_frame_r = np.linalg.norm(jaw_pos_r - jaw_right[:, :3], axis=1) if hand != "left" else np.zeros(N)

    # 2. 轨迹重合度
    err_overlap_l = np.zeros(N); err_overlap_r = np.zeros(N)
    if hand != "right":
        tree_l = cKDTree(jaw_left[:, :3])
        err_overlap_l, _ = tree_l.query(jaw_pos_l)
    if hand != "left":
        tree_r = cKDTree(jaw_right[:, :3])
        err_overlap_r, _ = tree_r.query(jaw_pos_r)

    # 3. 关键帧检测
    kf_mask = detect_keyframes(ep, N, freq, hand)
    n_kf = int(kf_mask.sum())

    # 打印结果
    ae_frame = np.maximum(err_frame_l, err_frame_r) if hand == "both" else (err_frame_r if hand == "right" else err_frame_l)
    ae_overlap = np.maximum(err_overlap_l, err_overlap_r) if hand == "both" else (err_overlap_r if hand == "right" else err_overlap_l)
    ae_ori = np.maximum(ori_err_l, ori_err_r) if hand == "both" else (ori_err_r if hand == "right" else ori_err_l)

    print("    逐帧误差:   R mean={:.4f} max={:.4f} | L mean={:.4f} max={:.4f}".format(
        err_frame_r.mean(), err_frame_r.max(), err_frame_l.mean(), err_frame_l.max()))
    print("    轨迹重合度: R mean={:.4f} max={:.4f} | L mean={:.4f} max={:.4f}".format(
        err_overlap_r.mean(), err_overlap_r.max(), err_overlap_l.mean(), err_overlap_l.max()))
    print("    朝向误差:   R mean={:.1f}° max={:.1f}° | L mean={:.1f}° max={:.1f}°".format(
        ori_err_r.mean(), ori_err_r.max(), ori_err_l.mean(), ori_err_l.max()))
    if n_kf > 2:
        print("    关键帧({}/{}): 逐帧={:.4f} 重合={:.4f} 朝向={:.1f}°".format(
            n_kf, N, ae_frame[kf_mask].mean(), ae_overlap[kf_mask].mean(), ae_ori[kf_mask].mean()))

    ep["optimized_joints"] = optimized
    ep["_base_joint_indices"] = robot_info["base_joint_indices"]
    ep["_actuated_names"] = robot_info["actuated_names"]
    ep["tracking_errors"] = {
        "left_frame": err_frame_l, "right_frame": err_frame_r,
        "left_overlap": err_overlap_l, "right_overlap": err_overlap_r,
        "left_ori": ori_err_l, "right_ori": ori_err_r,
        "keyframe_mask": kf_mask,
    }
    ep["tracking"] = {
        "pass_rate": float(np.mean(ae_frame < 0.01)),
        "max_error": float(ae_frame.max()),
        "mean_error": float(ae_frame.mean()),
        "overlap_max": float(ae_overlap.max()),
        "overlap_mean": float(ae_overlap.mean()),
        "ori_mean": float(ae_ori.mean()),
        "ori_max": float(ae_ori.max()),
        "keyframe_frame_mean": float(ae_frame[kf_mask].mean()) if n_kf > 0 else 0,
        "keyframe_overlap_mean": float(ae_overlap[kf_mask].mean()) if n_kf > 0 else 0,
        "keyframe_ori_mean": float(ae_ori[kf_mask].mean()) if n_kf > 0 else 0,
        "num_keyframes": n_kf,
    }
    return ep


# ===================================================================
# Step 5: 输出构建
