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

    # 声明一个变量用来存视频地址
    final_video_path = ""

    try:
        # 👇 接收录像模块传回来的视频完整路径
        final_video_path = camera_rec.setup()
        print("🚀 导弹发射，蛇形机动同步录制中...")

        start_time = time.time()

        for frame_id in range(TOTAL_FRAMES):
            loop_start = time.time()
            progress_t = frame_id / (TOTAL_FRAMES - 1)

            current_pose = missile_ctrl.update_pose(progress_t)

            current_time = time.time() - start_time
            camera_rec.record_frame(frame_id, current_time, current_pose)

            if frame_id % 15 == 0:
                print(f"▶️ 进度: {progress_t * 100:.1f}% | 帧: {frame_id}/{TOTAL_FRAMES} | 耗时: {current_time:.2f}s")

            elapsed = time.time() - loop_start
            sleep_time = RECORD_RATE - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("🎯 导弹抵达终点，任务完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户手动中止。")
    finally:
        if 'camera_rec' in locals():
            camera_rec.close()

        # 👇 在所有任务结束后，高亮打印视频的绝对路径！
        if final_video_path:
            print("-" * 50)
            print(f"🎬 杀青！本次大片已保存至:\n👉 {final_video_path}")
            print("-" * 50)

        print("🔌 仿真环境已断开连接。")


if __name__ == "__main__":
    main()