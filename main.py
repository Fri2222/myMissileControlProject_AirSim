import airsim
import time
import numpy as np
from scipy.spatial.transform import Rotation as R

# ================= 配置区域 =================
MISSILE_NAME = "Missile_1"  # 必须与 UE4 世界大纲里的名字完全一致
SPEED = 0.05  # 动作速度 (越小越慢，越流畅)
TOTAL_STEPS = 200  # 飞行的总帧数

# 定义坐标 (AirSim 使用 NED 坐标系: X前, Y右, Z下)
# 注意：Z 是负数代表“天上”
START_POS = np.array([0, 0, -2])  # 起点：原点上方-Z米
END_POS = np.array([150, 10, -2])  # 终点：前方X米，右边Y米，高度-Z米
CONTROL_POINT = np.array([25, 0, -50])  # 贝塞尔曲线控制点 (让它飞得高一点)


# ===========================================

def connect_ue4():
    """连接到 UE4"""
    print("正在连接 UE4 (AirSim)...")
    # 即使是 ComputerVision 模式，我们也用 VehicleClient
    client = airsim.VehicleClient()
    client.confirmConnection()
    return client


def calculate_bezier_point(t, p0, p1, p2):
    """
    数学公式：计算二阶贝塞尔曲线上的点
    t: 0到1之间的进度
    """
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2


def get_rotation_from_velocity(velocity):
    """
    让导弹头始终朝向飞行的方向
    """
    norm = np.linalg.norm(velocity)
    if norm < 1e-6:
        return airsim.Quaternionr(0, 0, 0, 1)

    direction = velocity / norm

    # 计算从默认方向 (X轴) 旋转到当前速度方向的旋转矩阵
    # 假设你的模型默认是红色箭头(X)朝前的
    rot = R.align_vectors([direction], [[1, 0, 0]])[0]
    quat = rot.as_quat()  # 返回 x, y, z, w

    return airsim.Quaternionr(quat[0], quat[1], quat[2], quat[3])


def main():
    client = connect_ue4()

    # 1. 验证是否找到了导弹
    try:
        pose = client.simGetObjectPose(MISSILE_NAME)
        print(f"✅ 成功锁定目标: {MISSILE_NAME}")
        print(f"   当前位置: X={pose.position.x_val:.1f}, Y={pose.position.y_val:.1f}, Z={pose.position.z_val:.1f}")
    except:
        print(f"❌ 错误：找不到名为 '{MISSILE_NAME}' 的物体！")
        print("   请检查：1. UE4是否在运行？ 2. 名字是否拼写正确？")
        return


    # 2. 开始飞行循环
    for i in range(TOTAL_STEPS):
        t = i / TOTAL_STEPS

        # A. 计算当前这一帧的位置
        current_pos = calculate_bezier_point(t, START_POS, CONTROL_POINT, END_POS)

        # B. 偷看下一帧的位置（用来计算朝向）
        next_t = (i + 1) / TOTAL_STEPS
        next_pos = calculate_bezier_point(next_t, START_POS, CONTROL_POINT, END_POS)

        # C. 计算速度方向
        velocity = next_pos - current_pos

        # D. 计算朝向 (四元数)
        orientation = get_rotation_from_velocity(velocity)

        # E. 组合成 Pose 对象
        pose = airsim.Pose(
            airsim.Vector3r(current_pos[0], current_pos[1], current_pos[2]),
            orientation
        )

        # F. 发送指令给 UE4 (核心步骤)
        # teleport=True 表示强制瞬移，忽略碰撞
        client.simSetObjectPose(MISSILE_NAME, pose, teleport=True)

        # 控制帧率
        time.sleep(SPEED)

    print("🎯 命中目标 (模拟结束)")


if __name__ == "__main__":
    main()