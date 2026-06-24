#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Retarget 数据 sim 回放 + 真机 replay。独立运行, 不依赖 R1Pro_core。

用法:
  # sim 回放 (需要 mujoco + DISPLAY)
  python3 -m fastumi_retarget.replay --episode dataset.hdf5 --sim

  # 真机 replay (需要 ROS2, 在机器人上运行)
  python3 -m fastumi_retarget.replay --episode dataset.hdf5 --real

  # 批量 sim
  python3 -m fastumi_retarget.replay --batch_dir ./retargeted/ --sim
"""
import argparse
import os
import sys
import time
import select

import numpy as np
import h5py

# --- MuJoCo ---
_MUJOCO_AVAILABLE = False
try:
    import mujoco
    import mujoco.viewer
    _MUJOCO_AVAILABLE = True
except ImportError:
    pass

# --- ROS2 (真机用, 可选) ---
_ROS_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    _ROS_AVAILABLE = True
except ImportError:
    pass


# R1Pro 简化 MJCF (无需 mesh 文件)
R1PRO_MJCF = """
<mujoco model="r1pro_full">
  <compiler angle="radian"/>
  <option gravity="0 0 -9.81" timestep="0.002"/>
  <worldbody>
    <body name="base_link" pos="0 0 0">
      <freejoint/>
      <geom type="cylinder" size="0.22 0.1" rgba="0.45 0.45 0.55 1"/>
      <body name="torso_link1" pos="-0.079032 0 0.34265">
        <joint name="torso_joint1" type="hinge" axis="0 1 0" range="-1.1345 1.8326"/>
        <geom type="cylinder" size="0.06 0.2" rgba="0.55 0.55 0.7 1"/>
        <body name="torso_link2" pos="0 0 0.4">
          <joint name="torso_joint2" type="hinge" axis="0 1 0" range="-2.7925 2.5307"/>
          <geom type="cylinder" size="0.05 0.22" rgba="0.55 0.55 0.7 1"/>
          <body name="torso_link3" pos="0 0.0001 0.3">
            <joint name="torso_joint3" type="hinge" axis="0 -1 0" range="-1.8326 1.5708"/>
            <geom type="cylinder" size="0.05 0.08" rgba="0.55 0.55 0.7 1"/>
            <body name="torso_link4" pos="0 -0.0001 0.09962">
              <joint name="torso_joint4" type="hinge" axis="0 0 1" range="-3.0543 3.0543"/>
              <geom type="cylinder" size="0.18 0.08" rgba="0.5 0.5 0.65 1"/>
              <body name="left_arm_base" pos="-0.00048618 0.097234 0.30302">
                <body name="left_arm_link1" pos="0 0.0735 0">
                  <joint name="left_arm_joint1" type="hinge" axis="0 1 0" range="-4.4506 1.3090"/>
                  <geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.025" rgba="0.82 0.82 1 1"/>
                  <body name="left_arm_link2" pos="0.025012 0.081265 0">
                    <joint name="left_arm_joint2" type="hinge" axis="1 0 0" range="-0.1745 3.1416"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.06" size="0.022" rgba="0.82 0.82 1 1"/>
                    <body name="left_arm_link3" pos="-0.025012 0 -0.1155">
                      <joint name="left_arm_joint3" type="hinge" axis="0 0 1" range="-2.3562 2.3562"/>
                      <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.022" rgba="0.82 0.82 1 1"/>
                      <body name="left_arm_link4" pos="0 0.035065 -0.1945">
                        <joint name="left_arm_joint4" type="hinge" axis="0 1 0" range="-2.0944 0.3491"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.07" size="0.018" rgba="0.82 0.82 1 1"/>
                        <body name="left_arm_link5" pos="0 -0.035065 -0.095">
                          <joint name="left_arm_joint5" type="hinge" axis="0 0 1" range="-2.3562 2.3562"/>
                          <geom type="capsule" fromto="0 0 0 0 0 -0.06" size="0.018" rgba="0.82 0.82 1 1"/>
                          <body name="left_arm_link6" pos="0 -0.028 -0.16305">
                            <joint name="left_arm_joint6" type="hinge" axis="0 1 0" range="-1.0472 1.0472"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.015" rgba="0.82 0.82 1 1"/>
                            <body name="left_arm_link7" pos="0.0295 0.028 0">
                              <joint name="left_arm_joint7" type="hinge" axis="1 0 0" range="-1.5708 1.5708"/>
                              <geom type="capsule" fromto="0 0 0 -0.025 0 -0.07" size="0.012" rgba="0.82 0.82 1 1"/>
                              <body name="left_gripper" pos="-0.0295 0 -0.16065">
                                <geom type="capsule" fromto="-0.01 0 0 0.01 0 0" size="0.012" rgba="0.85 0.75 0.75 1"/>
                                <body pos="0 0.013 -0.037">
                                  <joint name="left_grip1" type="slide" axis="0 1 0" range="0 0.05"/>
                                  <geom type="capsule" fromto="0 0 -0.01 0 0 0.01" size="0.005" rgba="0.82 0.82 1 1"/>
                                </body>
                                <body pos="0 -0.013 -0.037">
                                  <joint name="left_grip2" type="slide" axis="0 1 0" range="-0.05 0"/>
                                  <geom type="capsule" fromto="0 0 -0.01 0 0 0.01" size="0.005" rgba="0.82 0.82 1 1"/>
                                </body>
                              </body>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
              <body name="right_arm_base" pos="-0.00048706 -0.097236 0.30302">
                <body name="right_arm_link1" pos="0 -0.0735 0">
                  <joint name="right_arm_joint1" type="hinge" axis="0 1 0" range="-4.4506 1.3090"/>
                  <geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.025" rgba="0.82 0.82 1 1"/>
                  <body name="right_arm_link2" pos="0.023988 -0.081265 0">
                    <joint name="right_arm_joint2" type="hinge" axis="1 0 0" range="-3.1416 0.1745"/>
                    <geom type="capsule" fromto="0 0 0 0 0 -0.06" size="0.022" rgba="0.82 0.82 1 1"/>
                    <body name="right_arm_link3" pos="-0.023988 0 -0.1155">
                      <joint name="right_arm_joint3" type="hinge" axis="0 0 1" range="-2.3562 2.3562"/>
                      <geom type="capsule" fromto="0 0 0 0 0 -0.1" size="0.022" rgba="0.82 0.82 1 1"/>
                      <body name="right_arm_link4" pos="0 0.035058 -0.1945">
                        <joint name="right_arm_joint4" type="hinge" axis="0 1 0" range="-2.0944 0.3491"/>
                        <geom type="capsule" fromto="0 0 0 0 0 -0.07" size="0.018" rgba="0.82 0.82 1 1"/>
                        <body name="right_arm_link5" pos="0 -0.035058 -0.095">
                          <joint name="right_arm_joint5" type="hinge" axis="0 0 1" range="-2.3562 2.3562"/>
                          <geom type="capsule" fromto="0 0 0 0 0 -0.06" size="0.018" rgba="0.82 0.82 1 1"/>
                          <body name="right_arm_link6" pos="0 0.028001 -0.16305">
                            <joint name="right_arm_joint6" type="hinge" axis="0 1 0" range="-1.0472 1.0472"/>
                            <geom type="capsule" fromto="0 0 0 0 0 -0.05" size="0.015" rgba="0.82 0.82 1 1"/>
                            <body name="right_arm_link7" pos="0.0295 -0.028001 0">
                              <joint name="right_arm_joint7" type="hinge" axis="1 0 0" range="-1.5708 1.5708"/>
                              <geom type="capsule" fromto="0 0 0 -0.025 0 -0.07" size="0.012" rgba="0.82 0.82 1 1"/>
                              <body name="right_gripper" pos="-0.0295 0 -0.16065">
                                <geom type="capsule" fromto="-0.01 0 0 0.01 0 0" size="0.012" rgba="0.85 0.75 0.75 1"/>
                                <body pos="0 0.013 -0.037">
                                  <joint name="right_grip1" type="slide" axis="0 1 0" range="0 0.05"/>
                                  <geom type="capsule" fromto="0 0 -0.01 0 0 0.01" size="0.005" rgba="0.82 0.82 1 1"/>
                                </body>
                                <body pos="0 -0.013 -0.037">
                                  <joint name="right_grip2" type="slide" axis="0 1 0" range="-0.05 0"/>
                                  <geom type="capsule" fromto="0 0 -0.01 0 0 0.01" size="0.005" rgba="0.82 0.82 1 1"/>
                                </body>
                              </body>
                            </body>
                          </body>
                        </body>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# qpos layout: freejoint(7) + torso(4) + left_arm(7) + left_grip(2) + right_arm(7) + right_grip(2) = 29
_FREE_QPOS = slice(0, 7)
_TORSO_QPOS = slice(7, 11)
_LEFT_ARM_QPOS = slice(11, 18)
_LEFT_GRIP_QPOS = slice(18, 20)
_RIGHT_ARM_QPOS = slice(20, 27)
_RIGHT_GRIP_QPOS = slice(27, 29)

# ROS2 topics
TOPIC_TARGET_TORSO = "/motion_target/target_joint_state_torso"
TOPIC_TARGET_ARM_LEFT = "/motion_target/target_joint_state_arm_left"
TOPIC_TARGET_ARM_RIGHT = "/motion_target/target_joint_state_arm_right"
TOPIC_TARGET_GRIP_LEFT = "/motion_target/target_position_gripper_left"
TOPIC_TARGET_GRIP_RIGHT = "/motion_target/target_position_gripper_right"
TOPIC_FB_TORSO = "/hdas/feedback_torso"
TOPIC_FB_ARM_LEFT = "/hdas/feedback_arm_left"
TOPIC_FB_ARM_RIGHT = "/hdas/feedback_arm_right"

TORSO_VEL = [1.5, 1.5, 1.5, 5.0]
ARM_VEL = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]


def load_episode(filepath, demo_index="0"):
    """加载 processed dataset 或原始 episode。"""
    with h5py.File(str(filepath), "r") as f:
        if "data" in f:
            freq = float(f.attrs.get("freq_hz", 30))
            dt = 1.0 / freq
            demo_key = "data/demo_{}".format(demo_index)
            if demo_key not in f:
                available = [k for k in f["data"].keys() if k.startswith("demo_")]
                print("demo_{} 不存在。可用: {}".format(demo_index, available))
                return None
            demo = f[demo_key]
            joint_pos = demo["actions"][:]
            base_pos = demo["base_actions"][:] if "base_actions" in demo else None
            if "move_to_start" in demo:
                mts = demo["move_to_start/joints"][:]
                joint_pos = np.concatenate([mts, joint_pos], axis=0)
                if base_pos is not None:
                    mts_b = demo["move_to_start/base"][:] if "move_to_start/base" in demo else np.zeros((len(mts), 3))
                    base_pos = np.concatenate([mts_b, base_pos], axis=0)
            timestamps = np.arange(len(joint_pos)) * dt
            grade = str(f.attrs.get("grade", ""))
            score = float(f.attrs.get("quality_score", 0))
            return {
                "joint_pos": joint_pos, "timestamps": timestamps,
                "base_pos": base_pos, "freq": freq,
                "format": "processed", "grade": grade, "score": score,
            }
        if "action" in f:
            jp = f["action"][:]
            fmt = "sim"
        else:
            jp = f["observations/joint_positions"][:]
            fmt = "recorder"
        ts = f["timestamps"][:]
        freq = float(f.attrs.get("record_freq_hz", 30))
        return {
            "joint_pos": jp, "timestamps": ts,
            "base_pos": None, "freq": freq, "format": fmt,
            "grade": "", "score": 0,
        }


def frame_to_qpos(frame, fmt, nq, base_frame=None):
    """关节数据 → MuJoCo qpos。"""
    qpos = np.zeros(nq, dtype=np.float64)
    qpos[3] = 1.0  # freejoint quat w=1
    if base_frame is not None:
        qpos[0] = base_frame[0]
        qpos[1] = base_frame[1]
        # yaw → quat
        yaw = base_frame[2] if len(base_frame) > 2 else 0
        qpos[3] = np.cos(yaw / 2)
        qpos[6] = np.sin(yaw / 2)
    if fmt == "processed":
        qpos[_TORSO_QPOS] = frame[0:4]
        qpos[_LEFT_ARM_QPOS] = frame[4:11]
        qpos[_RIGHT_ARM_QPOS] = frame[11:18]
        gl = frame[18] if len(frame) > 18 else 0.0
        gr = frame[19] if len(frame) > 19 else 0.0
        qpos[_LEFT_GRIP_QPOS] = [gl * 0.05, -gl * 0.05]
        qpos[_RIGHT_GRIP_QPOS] = [gr * 0.05, -gr * 0.05]
    elif fmt == "sim":
        qpos[_TORSO_QPOS] = frame[0:4]
        qpos[_LEFT_ARM_QPOS] = frame[4:11]
        gl = frame[11]
        qpos[_LEFT_GRIP_QPOS] = [gl * 0.05, -gl * 0.05]
        qpos[_RIGHT_ARM_QPOS] = frame[12:19]
        gr = frame[19]
        qpos[_RIGHT_GRIP_QPOS] = [gr * 0.05, -gr * 0.05]
    else:
        qpos[_TORSO_QPOS] = frame[0:4]
        qpos[_LEFT_ARM_QPOS] = frame[4:11]
        qpos[_RIGHT_ARM_QPOS] = frame[11:18]
        gl = (frame[18] / 100.0) if len(frame) > 18 else 0.0
        gr = (frame[19] / 100.0) if len(frame) > 19 else 0.0
        qpos[_LEFT_GRIP_QPOS] = [gl * 0.05, -gl * 0.05]
        qpos[_RIGHT_GRIP_QPOS] = [gr * 0.05, -gr * 0.05]
    return qpos


def replay_sim(ep_data, speed=1.0, auto=False):
    """MuJoCo sim 回放。"""
    if not _MUJOCO_AVAILABLE:
        print("mujoco 未安装: pip install mujoco", file=sys.stderr)
        return 1

    joint_pos = ep_data["joint_pos"]
    timestamps = ep_data["timestamps"]
    base_pos = ep_data["base_pos"]
    fmt = ep_data["format"]
    freq = ep_data["freq"]
    N = len(joint_pos)
    dt_frames = np.diff(timestamps)
    dt_frames = np.clip(dt_frames, 1e-4, 1.0)

    model = mujoco.MjModel.from_xml_string(R1PRO_MJCF)
    data = mujoco.MjData(model)

    space_pressed = [False]
    def key_cb(keycode):
        if keycode == 32:
            space_pressed[0] = True

    try:
        viewer_ctx = mujoco.viewer.launch_passive(
            model=model, data=data, show_left_ui=False, show_right_ui=False,
            key_callback=key_cb)
    except Exception as e:
        print("MuJoCo viewer 启动失败: {}".format(e))
        return 1

    grade = ep_data.get("grade", "")
    score = ep_data.get("score", 0)
    if grade:
        print("  质量: {} (score={:.2f})".format(grade, score))

    if not auto:
        print("\n按空格或 Enter 开始 (speed={:.1f}x)...".format(speed))

    with viewer_ctx as viewer:
        if not auto:
            while viewer.is_running():
                if space_pressed[0]:
                    break
                if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                    sys.stdin.readline()
                    break
                viewer.sync()
            if not viewer.is_running():
                return 0

        space_pressed[0] = False
        try:
            for i in range(N):
                if not viewer.is_running():
                    break
                if space_pressed[0]:
                    space_pressed[0] = False
                    print("  暂停。按空格继续...")
                    while viewer.is_running() and not space_pressed[0]:
                        viewer.sync()
                        time.sleep(0.05)
                    space_pressed[0] = False
                    if not viewer.is_running():
                        break

                t_start = time.monotonic()
                bf = base_pos[i] if base_pos is not None and i < len(base_pos) else None
                data.qpos[:] = frame_to_qpos(joint_pos[i], fmt, model.nq, base_frame=bf)
                mujoco.mj_forward(model, data)
                viewer.sync()

                if i < N - 1:
                    wait = dt_frames[i] / speed
                    elapsed = time.monotonic() - t_start
                    if elapsed < wait:
                        time.sleep(wait - elapsed)

                if (i + 1) % (int(freq) * 5) == 0 or i == N - 1:
                    print("  {}/{} ({:.1f}s)".format(i + 1, N, (i + 1) / freq))

            print("回放完成。")
            if not auto:
                print("关闭窗口退出。")
                while viewer.is_running():
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n中断。")
    return 0


def replay_real(ep_data, speed=1.0):
    """真机 replay (ROS2, 无 R1Pro_core)。"""
    if not _ROS_AVAILABLE:
        print("ROS2 未安装, 无法真机 replay。", file=sys.stderr)
        return 1

    joint_pos = ep_data["joint_pos"]
    fmt = ep_data["format"]
    freq = ep_data["freq"]
    N = len(joint_pos)
    timestamps = ep_data["timestamps"]
    dt_frames = np.diff(timestamps)
    dt_frames = np.clip(dt_frames, 1e-4, 1.0)

    # 夹爪归一化 → mm
    if fmt in ("processed", "sim"):
        grip_l = np.clip(joint_pos[:, 18], 0, 1) * 100 if joint_pos.shape[1] > 18 else np.zeros(N)
        grip_r = np.clip(joint_pos[:, 19], 0, 1) * 100 if joint_pos.shape[1] > 19 else np.zeros(N)
    else:
        grip_l = joint_pos[:, 18] if joint_pos.shape[1] > 18 else np.zeros(N)
        grip_r = joint_pos[:, 19] if joint_pos.shape[1] > 19 else np.zeros(N)

    rclpy.init()
    node = Node("retarget_replay")

    pub_torso = node.create_publisher(JointState, TOPIC_TARGET_TORSO, 10)
    pub_arm_l = node.create_publisher(JointState, TOPIC_TARGET_ARM_LEFT, 10)
    pub_arm_r = node.create_publisher(JointState, TOPIC_TARGET_ARM_RIGHT, 10)
    pub_grip_l = node.create_publisher(JointState, TOPIC_TARGET_GRIP_LEFT, 10)
    pub_grip_r = node.create_publisher(JointState, TOPIC_TARGET_GRIP_RIGHT, 10)

    def send_frame(i):
        f = joint_pos[i]
        stamp = node.get_clock().now().to_msg()

        msg_t = JointState()
        msg_t.header.stamp = stamp
        msg_t.position = [float(v) for v in f[0:4]]
        msg_t.velocity = [float(v) for v in TORSO_VEL]
        pub_torso.publish(msg_t)

        msg_l = JointState()
        msg_l.header.stamp = stamp
        msg_l.position = [float(v) for v in f[4:11]]
        msg_l.velocity = [float(v) for v in ARM_VEL]
        pub_arm_l.publish(msg_l)

        msg_r = JointState()
        msg_r.header.stamp = stamp
        msg_r.position = [float(v) for v in f[11:18]]
        msg_r.velocity = [float(v) for v in ARM_VEL]
        pub_arm_r.publish(msg_r)

        gl = JointState()
        gl.header.stamp = stamp
        gl.position = [float(grip_l[i])]
        pub_grip_l.publish(gl)

        gr = JointState()
        gr.header.stamp = stamp
        gr.position = [float(grip_r[i])]
        pub_grip_r.publish(gr)

    print("过渡到起始位姿...")
    for _ in range(90):
        send_frame(0)
        rclpy.spin_once(node, timeout_sec=0)
        time.sleep(1 / 30)

    print("回放中... ({} 帧, speed={:.1f}x)".format(N, speed))
    try:
        for i in range(N):
            t_start = time.monotonic()
            send_frame(i)
            rclpy.spin_once(node, timeout_sec=0)

            if i < N - 1:
                wait = dt_frames[i] / speed
                elapsed = time.monotonic() - t_start
                if elapsed < wait:
                    time.sleep(wait - elapsed)

            if (i + 1) % (int(freq) * 5) == 0 or i == N - 1:
                print("  {}/{} ({:.1f}s)".format(i + 1, N, (i + 1) / freq))
    except KeyboardInterrupt:
        print("\n中断。")

    print("回放完成。")
    node.destroy_node()
    rclpy.shutdown()
    return 0


def main():
    parser = argparse.ArgumentParser(description="Retarget 数据回放 (sim + 真机)")
    parser.add_argument("--episode", type=str, help="单条 dataset.hdf5")
    parser.add_argument("--batch_dir", type=str, help="批量目录")
    parser.add_argument("--demo", type=str, default="0", help="demo index")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--sim", action="store_true", help="MuJoCo sim 回放")
    parser.add_argument("--real", action="store_true", help="真机 replay (ROS2)")
    parser.add_argument("--auto", action="store_true", help="自动开始")
    args = parser.parse_args()

    if not args.episode and not args.batch_dir:
        parser.print_help()
        return 1

    if not args.sim and not args.real:
        args.sim = True

    import glob
    episodes = []
    if args.batch_dir:
        episodes = sorted(glob.glob(os.path.join(args.batch_dir, "**", "dataset.hdf5"), recursive=True))
        if not episodes:
            print("未找到 dataset.hdf5: {}".format(args.batch_dir))
            return 1
        print("批量回放: {} 条".format(len(episodes)))
        args.auto = True
    else:
        episodes = [args.episode]

    for ep_path in episodes:
        print("\n加载: {}".format(ep_path))
        ep_data = load_episode(ep_path, demo_index=args.demo)
        if ep_data is None:
            continue
        jp = ep_data["joint_pos"]
        print("  {} 帧, {:.1f}Hz, {:.1f}s, 格式={}".format(
            len(jp), ep_data["freq"], len(jp) / ep_data["freq"], ep_data["format"]))

        if args.sim:
            replay_sim(ep_data, speed=args.speed, auto=args.auto)
        elif args.real:
            replay_real(ep_data, speed=args.speed)

    return 0


if __name__ == "__main__":
    sys.exit(main())
