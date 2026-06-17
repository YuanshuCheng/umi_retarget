"""URDF 加载: 解析 URDF → pyroki Robot + 关节/link 索引映射。"""
import os
import tempfile
import xml.etree.ElementTree as ET

import pyroki as pk


def load_pyroki_robot(urdf_path):
    """加载 URDF 到 pyroki, 返回 robot 和关键 link/joint 索引。"""
    import yourdfpy

    tree = ET.parse(urdf_path)
    root = tree.getroot()
    keep_prefixes = ("base_link", "torso_", "left_arm_", "right_arm_",
                     "left_gripper", "right_gripper")
    to_remove = []
    for elem in root:
        name = elem.get("name", "")
        if elem.tag == "link":
            if name != "base_link" and not any(name.startswith(p) for p in keep_prefixes):
                to_remove.append(elem)
        elif elem.tag == "joint":
            child = elem.find("child")
            parent = elem.find("parent")
            cl = child.get("link", "") if child is not None else ""
            pl = parent.get("link", "") if parent is not None else ""
            if not (any(cl.startswith(p) for p in keep_prefixes) and
                    any(pl.startswith(p) for p in keep_prefixes)):
                to_remove.append(elem)
    for elem in set(to_remove):
        root.remove(elem)

    wb = root.find(".")
    for name in ["chassis_x_link", "chassis_y_link", "chassis_yaw_link"]:
        ET.SubElement(wb, "link", name=name)

    chassis_joints = [
        ("chassis_x_joint", "prismatic", "1 0 0", "chassis_x_link", "chassis_y_link", "-10 10", "1.0"),
        ("chassis_y_joint", "prismatic", "0 1 0", "chassis_y_link", "chassis_yaw_link", "-10 10", "1.0"),
        ("chassis_yaw_joint", "revolute", "0 0 1", "chassis_yaw_link", "base_link", "-6.2832 6.2832", "3.0"),
    ]
    for jname, jtype, axis, parent_l, child_l, limits, vel in chassis_joints:
        j = ET.SubElement(wb, "joint", name=jname, type=jtype)
        ET.SubElement(j, "parent", link=parent_l)
        ET.SubElement(j, "child", link=child_l)
        ET.SubElement(j, "axis", xyz=axis)
        lo, hi = limits.split()
        ET.SubElement(j, "limit", lower=lo, upper=hi, velocity=vel, effort="100")

    tmp = tempfile.NamedTemporaryFile(suffix=".urdf", delete=False, mode="w")
    tmp.write(ET.tostring(root, encoding="unicode"))
    tmp.close()

    urdf = yourdfpy.URDF.load(
        tmp.name,
        build_scene_graph=False, build_collision_scene_graph=False,
        load_meshes=False, load_collision_meshes=False,
    )
    os.unlink(tmp.name)

    if urdf.base_link is None:
        all_links = set(l.name for l in urdf.robot.links)
        child_links = set(j.child for j in urdf.robot.joints)
        root_links = all_links - child_links
        if root_links:
            urdf._base_link = root_links.pop()
            print("    手动设置 base_link: {}".format(urdf._base_link))

    robot = pk.Robot.from_urdf(urdf)
    joint_names = list(robot.joints.actuated_names)
    link_names = list(robot.links.names)

    def _find_act(name):
        for i, n in enumerate(joint_names):
            if name == n:
                return i
        return -1

    def _find_link(name):
        for i, n in enumerate(link_names):
            if name == n:
                return i
        return -1

    left_arm_idxs = [_find_act("left_arm_joint{}".format(i + 1)) for i in range(7)]
    left_grip_idxs = [_find_act("left_gripper_finger_joint{}".format(i + 1)) for i in range(2)]
    right_arm_idxs = [_find_act("right_arm_joint{}".format(i + 1)) for i in range(7)]
    right_grip_idxs = [_find_act("right_gripper_finger_joint{}".format(i + 1)) for i in range(2)]
    base_joint_idxs = [_find_act("chassis_{}_joint".format(a)) for a in ["x", "y", "yaw"]]

    robot_coll = None
    try:
        from pyroki.collision import RobotCollision
        sphere_decomp = {
            "torso_link2": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.15]},
            "torso_link4": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.13]},
            "left_arm_link4": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.06]},
            "left_arm_link6": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.06]},
            "right_arm_link4": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.06]},
            "right_arm_link6": {"centers": [[0.0, 0.0, 0.0]], "radii": [0.06]},
        }
        ignore_pairs = (
            ("torso_link2", "torso_link4"),
            ("left_arm_link4", "left_arm_link6"),
            ("right_arm_link4", "right_arm_link6"),
            ("left_arm_link4", "right_arm_link4"),
            ("left_arm_link4", "right_arm_link6"),
            ("left_arm_link6", "right_arm_link4"),
            ("left_arm_link6", "right_arm_link6"),
        )
        robot_coll = RobotCollision.from_sphere_decomposition(
            sphere_decomp, urdf,
            user_ignore_pairs=ignore_pairs,
            ignore_immediate_adjacents=True)
        print("    自碰撞: {} 对碰撞检测".format(len(robot_coll.active_idx_i)))
    except Exception as e:
        print("    自碰撞加载失败: {}".format(e))

    info = {
        "robot": robot,
        "robot_coll": robot_coll,
        "left_ee_idx": _find_link("left_gripper_link"),
        "right_ee_idx": _find_link("right_gripper_link"),
        "torso_joint_indices": [_find_act("torso_joint{}".format(i + 1)) for i in range(4)],
        "left_elbow_idx": _find_act("left_arm_joint4"),
        "right_elbow_idx": _find_act("right_arm_joint4"),
        "left_arm_indices": [i for i in left_arm_idxs + left_grip_idxs if i >= 0],
        "right_arm_indices": [i for i in right_arm_idxs + right_grip_idxs if i >= 0],
        "base_joint_indices": [i for i in base_joint_idxs if i >= 0],
        "n_actuated": robot.joints.num_actuated_joints,
        "actuated_names": joint_names,
    }

    print("    pyroki robot: {} actuated joints, {} links".format(
        info["n_actuated"], len(link_names)))
    print("    actuated: {}".format(joint_names))
    print("    left_ee={} right_ee={}".format(info["left_ee_idx"], info["right_ee_idx"]))
    print("    torso={} elbows=[{},{}] base={}".format(
        info["torso_joint_indices"], info["left_elbow_idx"], info["right_elbow_idx"],
        info["base_joint_indices"]))

    return info
