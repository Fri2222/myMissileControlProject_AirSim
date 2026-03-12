import airsim
import time
from trajectory_control import MissileController
from camera_control import CameraRecorder

# ================= ⚙️ 全局核心配置 =================
MISSILE_NAME = "Missile_1"
SAVE_DIR = "H:/Missile_Video_Dataset"

TOTAL_FRAMES = 180  # 总帧数 (录制 3 秒)
RECORD_RATE = 0.0167  # 帧间隔 ≈ 60 FPS

# --- 🚀 坐标体系配置 (X前, Y右, Z下(负代表天空)) ---
# 1. 导弹飞行轨迹规划
START_POS = [0, 0, -2]  # 起点
SHARED_TARGET_POS = [50, 10, -30]  # 🎯 核心交汇点：导弹必定穿过此点，相机死死盯住此点！
END_POS = [150, 0, -50]  # 终点

# 2. 摄像机固定机位 (位于起点侧后方)
FIXED_CAMERA_POS = [-20, 20, -10]


# =================================================

def main():
    print("🔌 正在连接 UE4...")
    client = airsim.VehicleClient()
    client.confirmConnection()

    # 1. 初始化独立的模块
    missile_ctrl = MissileController(
        client=client,
        missile_name=MISSILE_NAME,
        start_pos=START_POS,
        target_pos=SHARED_TARGET_POS,
        end_pos=END_POS
    )

    camera_rec = CameraRecorder(
        client=client,
        camera_name="0",
        save_dir=SAVE_DIR,
        record_rate=RECORD_RATE,
        camera_pos=FIXED_CAMERA_POS,
        look_at_pos=SHARED_TARGET_POS
    )

    try:
        # 2. 启动录像环境
        camera_rec.setup()
        print("🚀 导弹发射，同步录制中...")

        start_time = time.time()

        # 3. 中控主循环：以帧驱动，保证录像和轨迹的绝对同步
        for frame_id in range(TOTAL_FRAMES):
            loop_start = time.time()
            progress_t = frame_id / (TOTAL_FRAMES - 1)  # 进度从 0 到 1

            # 👉 模块 A 工作：更新导弹位置并获取状态
            current_pose = missile_ctrl.update_pose(progress_t)

            # 👉 模块 B 工作：相机拍照并记录状态
            current_time = time.time() - start_time
            camera_rec.record_frame(frame_id, current_time, current_pose)

            if frame_id % 15 == 0:
                print(f"▶️ 进度: {progress_t * 100:.1f}% | 帧: {frame_id}/{TOTAL_FRAMES} | 耗时: {current_time:.2f}s")

            # 帧率控制
            elapsed = time.time() - loop_start
            sleep_time = RECORD_RATE - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("🎯 导弹抵达终点，任务完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户手动中止。")
    finally:
        # 4. 释放资源
        camera_rec.close()
        client.reset()
        client.enableApiControl(False)
        print("🔌 仿真环境已重置。")


if __name__ == "__main__":
    main()