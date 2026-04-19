import airsim
import numpy as np
import math
import random
from scipy.spatial.transform import Rotation as R

class MissileController:
    def __init__(self, client, missile_name, start_pos, end_pos, launch_delay=0.0, flight_duration=3.0,
                 control_point=None):
        self.client = client
        self.missile_name = missile_name
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)

        # ⏱️ 严格时间控制：强制拦截，最大延迟绝对不超过 3.0 秒
        self.launch_delay = min(float(launch_delay), 3.0)
        self.flight_duration = flight_duration

        # 🎯 自由飞行轨迹：完全抛弃之前的公共交汇点！
        if control_point is not None:
            self.control_point = np.array(control_point)
        else:
            # 自动在起点和终点之间生成一个随机的高空控制点
            # 每枚导弹会有自己独一无二的抛物线高度和左右偏移
            mid_point = (self.start_pos + self.end_pos) / 2.0
            random_offset = np.array([
                random.uniform(-30, 30),  # X轴：前后随机偏移
                random.uniform(-50, 50),  # Y轴：左右随机飘逸
                random.uniform(-80, -20)  # Z轴：抛物线高度随机（负数代表往天上飞）
            ])
            self.control_point = mid_point + random_offset

        # ================== 🎲 随机机动生成器 ==================
        self.time_warp_amp = random.uniform(-0.15, 0.15)
        self.wobble_amp1 = np.array([random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-5, 5)])
        self.wobble_amp2 = np.array([random.uniform(-4, 4), random.uniform(-4, 4), random.uniform(-2, 2)])

        print(f"🚀 [{self.missile_name}] 准备就绪 | 延迟起飞: {self.launch_delay:.2f}s | 飞行时间: {self.flight_duration:.2f}s")

    def _calculate_actual_pos(self, p):
        """核心数学引擎：计算加入扰动后的真实空间坐标"""
        t = p + self.time_warp_amp * math.sin(2 * math.pi * p)
        base_pos = (1 - t) ** 2 * self.start_pos + 2 * (1 - t) * t * self.control_point + t ** 2 * self.end_pos
        offset = self.wobble_amp1 * math.sin(2 * math.pi * p) + self.wobble_amp2 * math.sin(4 * math.pi * p)
        return base_pos + offset

    def _get_rotation_from_velocity(self, velocity):
        """让导弹头始终朝向飞行的瞬时方向"""
        norm = np.linalg.norm(velocity)
        if norm < 1e-6:
            return airsim.Quaternionr(0, 0, 0, 1)

        direction = velocity / norm
        rot = R.align_vectors([direction], [[1, 0, 0]])[0]
        quat = rot.as_quat()
        return airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])

    def update_pose(self, global_time):
        """根据仿真世界的绝对时间 (秒)，动态计算并更新位姿"""
        # 1. 把全局时间映射为进度 p (0~1)
        if global_time < self.launch_delay:
            p = 0.0  # 还没到这枚导弹的发射时间，死死锁在起点
        else:
            p = (global_time - self.launch_delay) / self.flight_duration
            p = min(p, 1.0)  # 到了终点就停在终点

        # 2. 计算当前位置
        current_pos = self._calculate_actual_pos(p)

        # 3. 计算切线方向(朝向)
        next_p = min(p + 0.01, 1.0)
        next_pos = self._calculate_actual_pos(next_p)
        velocity = next_pos - current_pos
        orientation = self._get_rotation_from_velocity(velocity)

        pose = airsim.Pose(
            airsim.Vector3r(current_pos[0], current_pos[1], current_pos[2]),
            orientation
        )
        self.client.simSetObjectPose(self.missile_name, pose, teleport=True)
        return pose