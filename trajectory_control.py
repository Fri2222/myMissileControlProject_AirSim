import airsim
import numpy as np
from scipy.spatial.transform import Rotation as R


class MissileController:
    def __init__(self, client, missile_name, start_pos, target_pos, end_pos):
        self.client = client
        self.missile_name = missile_name
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.end_pos = np.array(end_pos)

        # 核心数学：反向推导二阶贝塞尔控制点，确保 t=0.5 时精确穿过 target_pos
        self.control_point = 2 * self.target_pos - 0.5 * self.start_pos - 0.5 * self.end_pos

    def _calculate_bezier_point(self, t):
        """计算二阶贝塞尔曲线上的点"""
        return (1 - t) ** 2 * self.start_pos + 2 * (1 - t) * t * self.control_point + t ** 2 * self.end_pos

    def _get_rotation_from_velocity(self, velocity):
        """让导弹头始终朝向飞行方向"""
        norm = np.linalg.norm(velocity)
        if norm < 1e-6:
            return airsim.Quaternionr(0, 0, 0, 1)

        direction = velocity / norm
        rot = R.align_vectors([direction], [[1, 0, 0]])[0]
        quat = rot.as_quat()
        return airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])

    def update_pose(self, progress_t):
        """
        根据进度 t (0.0 到 1.0) 更新导弹位置并返回当前 pose
        """
        # 计算当前帧和下一帧的位置（用于算朝向）
        current_pos = self._calculate_bezier_point(progress_t)
        next_t = min(progress_t + 0.01, 1.0)
        next_pos = self._calculate_bezier_point(next_t)

        velocity = next_pos - current_pos
        orientation = self._get_rotation_from_velocity(velocity)

        # 组合 Pose 并发送给 UE4
        pose = airsim.Pose(
            airsim.Vector3r(current_pos[0], current_pos[1], current_pos[2]),
            orientation
        )
        self.client.simSetObjectPose(self.missile_name, pose, teleport=True)
        return pose