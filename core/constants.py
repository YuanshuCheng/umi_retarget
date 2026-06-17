"""常量和位姿工具函数。"""
import numpy as np

# FastUMI SLAM → Robot 世界系
FASTUMI_TO_WORLD = np.array([
    [ 0,  0,  1],
    [-1,  0,  0],
    [ 0, -1,  0],
], dtype=np.float64)

SHOULDER_HEIGHT = 1.445
ARM_COMFORTABLE_REACH = 0.47
GRIPPER_JAW_OFFSET = -0.065

FASTUMI_CAM_OFFSET = np.array([0.0, +0.07, +0.075])

INIT_POS_OFFSET_LEFT = np.array([0.0, +0.07, -0.07])
INIT_POS_OFFSET_RIGHT = np.array([0.0, -0.07, -0.07])

JOINT_LIMITS = np.array([
    (-1.1345, 1.8326), (-2.7925, 2.5307), (-1.8326, 1.5708), (-3.0543, 3.0543),
    (-4.4506, 1.3090), (-0.1745, 3.1416), (-2.3562, 2.3562),
    (-2.0944, 0.3491), (-2.3562, 2.3562), (-1.0472, 1.0472), (-1.5708, 1.5708),
    (-4.4506, 1.3090), (-3.1416, 0.1745), (-2.3562, 2.3562),
    (-2.0944, 0.3491), (-2.3562, 2.3562), (-1.0472, 1.0472), (-1.5708, 1.5708)])
MAX_VEL = np.array([0.5]*4 + [3.0]*7 + [3.0]*7)
MAX_ACC = np.array([5.0]*4 + [40.0]*7 + [40.0]*7)

GRIPPER_GRIP_MAX_FASTUMI = 87.0
GRIPPER_GRIP_MAX_R1PRO = 80.0
GRIP_TIGHTEN = 0.9


def quat_to_rotmat(q):
    qx, qy, qz, qw = q
    return np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qz*qw), 2*(qx*qz+qy*qw)],
        [2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2), 2*(qy*qz-qx*qw)],
        [2*(qx*qz-qy*qw), 2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]])


def rotmat_to_quat(R):
    qw = np.sqrt(max(0, 1+R[0,0]+R[1,1]+R[2,2])) / 2
    if abs(qw) >= 1e-6:
        qx=(R[2,1]-R[1,2])/(4*qw); qy=(R[0,2]-R[2,0])/(4*qw); qz=(R[1,0]-R[0,1])/(4*qw)
    else:
        qx = np.sqrt(max(0, 1+R[0,0]-R[1,1]-R[2,2])) / 2
        qy = np.sqrt(max(0, 1-R[0,0]+R[1,1]-R[2,2])) / 2
        qz = np.sqrt(max(0, 1-R[0,0]-R[1,1]+R[2,2])) / 2
        i = np.argmax([qx, qy, qz])
        if i==0: qy=(R[0,1]+R[1,0])/(4*qx); qz=(R[0,2]+R[2,0])/(4*qx)
        elif i==1: qx=(R[0,1]+R[1,0])/(4*qy); qz=(R[1,2]+R[2,1])/(4*qy)
        else: qx=(R[0,2]+R[2,0])/(4*qz); qy=(R[1,2]+R[2,1])/(4*qz)
    n = np.sqrt(qx**2+qy**2+qz**2+qw**2)
    return np.array([qx/n, qy/n, qz/n, qw/n])


def fastumi_pose_to_world(pos_f, quat_f):
    M = FASTUMI_TO_WORLD
    pos_w = M @ np.asarray(pos_f[:3], dtype=np.float64)
    R_f = quat_to_rotmat(quat_f)
    R_w = M @ R_f @ M.T
    return pos_w, rotmat_to_quat(R_w)
