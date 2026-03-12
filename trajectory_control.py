import airsim
import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R


class MissileController:
    def __init__(self, client, missile_name, start_pos, target_pos, end_pos):
        self.client = client
        self.missile_name = missile_name
        self.start_pos = np.array(start_pos)
        self.target_pos = np.array(target_pos)
        self.end_pos = np.array(end_pos)

        # 1. 基础轨迹控制点：反向推导二阶贝塞尔，确保中间时刻穿过 target_pos
        self.control_point = 2 * self.target_pos - 0.5 * self.start_pos - 0.5 * self.end_pos

        # ================== 🎲 随机机动生成器 ==================
        # 每次实例化导弹时，都会随机生成一套全新的飞行机动参数

        # 2. 变速参数 (时间扭曲 Time Warping)
        self.time_warp_amp = random.uniform(-0.15, 0.15)

        # 3. 变向参数 (空间蛇形走位 Swerving)
        self.wobble_amp1 = np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-5, 5)])
        self.wobble_amp2 = np.array([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-2, 2)])

        print(
            f"🎲 [轨迹生成] 变速因子: {self.time_warp_amp:.2f} | 最大变向机动: {np.linalg.norm(self.wobble_amp1):.1f}m")

    def _calculate_actual_pos(self, p):
        """核心数学引擎：包含时间扭曲(加减速)和空间偏移(变向)的复合轨迹"""
        t = p + self.time_warp_amp * math.sin(2 * math.pi * p)
        base_pos = (1 - t) ** 2 * self.start_pos + 2 * (1 - t) * t * self.control_point + t ** 2 * self.end_pos
        offset = self.wobble_amp1 * math.sin(2 * math.pi * p) + self.wobble_amp2 * math.sin(4 * math.pi * p)
        return base_pos + offset

    def _get_rotation_from_velocity(self, velocity):
        """让导弹头始终朝向飞行(机动)的瞬时方向"""
        norm = np.linalg.norm(velocity)
        if norm < 1e-6:
            return airsim.Quaternionr(0, 0, 0, 1)

        direction = velocity / norm
        rot = R.align_vectors([direction], [[1, 0, 0]])[0]
        quat = rot.as_quat()
        return airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])

    def update_pose(self, progress_p):
        """根据进度 p (0.0 到 1.0) 更新导弹位置并返回当前 pose"""
        current_pos = self._calculate_actual_pos(progress_p)
        next_p = min(progress_p + 0.01, 1.0)
        next_pos = self._calculate_actual_pos(next_p)

        velocity = next_pos - current_pos
        orientation = self._get_rotation_from_velocity(velocity)

        pose = airsim.Pose(
            airsim.Vector3r(current_pos[0], current_pos[1], current_pos[2]),
            orientation
        )
        self.client.simSetObjectPose(self.missile_name, pose, teleport=True)
        return pose