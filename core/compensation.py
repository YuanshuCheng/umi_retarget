"""Dynamics 补偿: GBT 残差预测 + pyroki 全局平滑。"""
import os
import pickle
import time

import numpy as np

N_JOINTS = 18
DEFAULT_FREQ = 30.0
POST_SMOOTH_CUTOFF = 3.0
SKIP_HEAD_FRAMES = 15


def build_features(q_j, freq=DEFAULT_FREQ):
    """构造单关节特征: [q, q_dot, q_ddot]。"""
    dt = 1.0 / freq
    q_dot = np.gradient(q_j, dt)
    q_ddot = np.gradient(q_dot, dt)
    return np.column_stack([q_j, q_dot, q_ddot])


def build_training_data(logs_data, freq=DEFAULT_FREQ, skip_head=SKIP_HEAD_FRAMES):
    """从所有 log 构造 per-joint 训练数据。等权重, 不区分关键帧。"""
    X_per_joint = [[] for _ in range(N_JOINTS)]
    y_per_joint = [[] for _ in range(N_JOINTS)]

    for ld in logs_data:
        q_cmd = ld["q_cmd"]
        q_actual = ld["q_actual"]
        N = len(q_cmd)

        start = min(skip_head, N // 4)
        if N - start < 10:
            continue

        for j in range(N_JOINTS):
            residual = q_cmd[start:, j] - q_actual[start:, j]
            feat = build_features(q_cmd[start:, j], freq)
            X_per_joint[j].append(feat)
            y_per_joint[j].append(residual)

    result_X, result_y = [], []
    for j in range(N_JOINTS):
        if X_per_joint[j]:
            result_X.append(np.concatenate(X_per_joint[j], axis=0))
            result_y.append(np.concatenate(y_per_joint[j], axis=0))
        else:
            result_X.append(np.zeros((0, 3)))
            result_y.append(np.zeros(0))

    return result_X, result_y


def train_residual_models(X_per_joint, y_per_joint, val_ratio=0.2):
    """训练 per-joint GBT 残差模型。所有帧等权重。"""
    from sklearn.ensemble import GradientBoostingRegressor

    models, train_rmse, val_rmse = [], [], []

    for j in range(N_JOINTS):
        X, y = X_per_joint[j], y_per_joint[j]

        if len(X) < 20:
            models.append(None)
            train_rmse.append(0.0)
            val_rmse.append(0.0)
            continue

        n_val = max(1, int(len(X) * val_ratio))
        indices = np.arange(len(X))
        np.random.seed(42 + j)
        np.random.shuffle(indices)

        X_train, y_train = X[indices[n_val:]], y[indices[n_val:]]
        X_val, y_val = X[indices[:n_val]], y[indices[:n_val]]

        model = GradientBoostingRegressor(
            n_estimators=200, max_depth=5, learning_rate=0.1,
            subsample=0.8, random_state=42,
        )
        model.fit(X_train, y_train)

        rmse_t = np.sqrt(np.mean((model.predict(X_train) - y_train) ** 2))
        rmse_v = np.sqrt(np.mean((model.predict(X_val) - y_val) ** 2))

        models.append(model)
        train_rmse.append(float(rmse_t))
        val_rmse.append(float(rmse_v))

    return models, train_rmse, val_rmse


def predict_residual(models, q_cmd, freq=DEFAULT_FREQ):
    """用 GBT 预测逐帧补偿量 (有噪声, 不直接使用)。"""
    N, n_j = q_cmd.shape
    compensation = np.zeros_like(q_cmd)
    for j in range(min(n_j, len(models))):
        if models[j] is None:
            continue
        feat = build_features(q_cmd[:, j], freq)
        compensation[:, j] = models[j].predict(feat)
    return compensation


# ===================================================================
# Step 2: pyroki 全局轨迹优化
# ===================================================================

def optimize_compensation_pyroki(q_cmd, residual_target, robot_info, cfg_weights,
                                 freq=DEFAULT_FREQ):
    """用 pyroki/jaxls 全局优化, 将 GBT 的带噪声预测变成平滑轨迹。

    决策变量: q_warped (N, n_actuated)
    Cost:
      - 关节角追踪: q_warped 接近 q_cmd + residual_target (等权重)
      - 平滑: velocity + acceleration
      - 关节限位
    初始化: q_cmd + residual_target (warm-start)
    """
    import jax.numpy as jnp
    import jaxls
    import pyroki as pk

    robot = robot_info["robot"]
    n_act = robot_info["n_actuated"]
    N = len(q_cmd)

    # 映射 18 维关节到 pyroki actuated 索引
    act_names = robot_info["actuated_names"]
    def _find_act(name):
        for i, n in enumerate(act_names):
            if n == name:
                return i
        return -1

    joint_map = []
    torso_names = ["torso_joint{}".format(i+1) for i in range(4)]
    left_names = ["left_arm_joint{}".format(i+1) for i in range(7)]
    right_names = ["right_arm_joint{}".format(i+1) for i in range(7)]
    for name in torso_names + left_names + right_names:
        joint_map.append(_find_act(name))
    active_idxs = jnp.array([i for i in joint_map if i >= 0])
    n_active = len(active_idxs)

    # q_target: GBT 预测的目标 (N, n_active)
    q_target_18 = q_cmd + residual_target
    q_target_full = np.zeros((N, n_act))
    q_cmd_full = np.zeros((N, n_act))
    for j18 in range(N_JOINTS):
        idx = joint_map[j18]
        if idx >= 0:
            q_target_full[:, idx] = q_target_18[:, j18]
            q_cmd_full[:, idx] = q_cmd[:, j18]

    q_target_jnp = jnp.array(q_target_full.astype(np.float32))
    init_cfg = jnp.array(q_target_full.astype(np.float32))

    # 决策变量
    traj_vars = robot.joint_var_cls(jnp.arange(N))
    dt = 1.0 / freq

    costs = []

    # 1. 关节角追踪 cost (等权重) — 用 rest_cost 实现 per-frame 追踪
    track_w = cfg_weights.get("track_weight", 20.0)
    costs.append(pk.costs.rest_cost(
        traj_vars, q_target_jnp, jnp.array([track_w])[None]))

    # 2. 平滑 cost (velocity)
    smooth_w = cfg_weights.get("smooth_weight", 5.0)
    if smooth_w > 0 and N > 1:
        costs.append(pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, N)),
            robot.joint_var_cls(jnp.arange(0, N - 1)),
            jnp.array([smooth_w])[None]))

    # 3. 平滑 cost (acceleration)
    acc_w = cfg_weights.get("acc_weight", 5.0)
    if acc_w > 0 and N > 4:
        costs.append(pk.costs.five_point_acceleration_cost(
            robot.joint_var_cls(jnp.arange(2, N - 2)),
            robot.joint_var_cls(jnp.arange(4, N)),
            robot.joint_var_cls(jnp.arange(3, N - 1)),
            robot.joint_var_cls(jnp.arange(1, N - 3)),
            robot.joint_var_cls(jnp.arange(0, N - 4)),
            dt, jnp.array([acc_w])[None]))

    # 4. 关节限位 cost
    import jax
    limit_w = cfg_weights.get("limit_weight", 100.0)
    if limit_w > 0:
        robot_batched = jax.tree.map(lambda x: jnp.broadcast_to(x, (N, *x.shape)), robot)
        costs.append(pk.costs.limit_cost(robot_batched, traj_vars,
                                         jnp.array([limit_w])[None]))

    # 求解
    num_iter = cfg_weights.get("num_iterations", 60)
    print("    pyroki 求解: {} 帧, {} costs, {} 迭代...".format(N, len(costs), num_iter))

    t0 = time.monotonic()
    problem = jaxls.LeastSquaresProblem(costs=costs, variables=[traj_vars]).analyze()
    print("    analyze (JIT): {:.1f}s".format(time.monotonic() - t0))

    t1 = time.monotonic()
    solution = problem.solve(
        initial_vals=jaxls.VarValues.make((traj_vars.with_value(init_cfg),)),
        termination=jaxls.TerminationConfig(max_iterations=num_iter),
    )
    print("    solve: {:.1f}s".format(time.monotonic() - t1))

    optimized = np.array(solution[traj_vars])

    # 提取 18 维关节角
    q_warped = np.zeros((N, N_JOINTS))
    for j18 in range(N_JOINTS):
        idx = joint_map[j18]
        if idx >= 0:
            q_warped[:, j18] = optimized[:, idx]

    return q_warped


# ===================================================================
# 完整补偿 pipeline
# ===================================================================

def compensate_trajectory(q_cmd, calib, robot_info, freq=DEFAULT_FREQ,
                          cfg_weights=None):
    """完整补偿: GBT 预测 + pyroki 全局平滑。"""
    if cfg_weights is None:
        cfg_weights = {}

    # Step 1: GBT 预测逐帧补偿目标
    residual_target = predict_residual(calib["models"], q_cmd, freq)

    # Step 2: pyroki 全局优化
    q_warped = optimize_compensation_pyroki(
        q_cmd, residual_target, robot_info, cfg_weights, freq)

    return q_warped
