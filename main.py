import airsim
import time
from trajectory_control import MissileController
from camera_control import CameraRecorder

# ================= ⚙️ 蜂群(多导弹)全局配置 =================
SAVE_DIR = "H:/Missile_Video_Dataset"

TOTAL_FRAMES = 180  # 总帧数 (录制 3 秒)
RECORD_RATE = 0.0167  # ≈ 60 FPS

# 🎯 核心交汇点：所有导弹都必须在 t=0.5 时穿过这个死机位中心点！
SHARED_TARGET_POS = [50, 10, -30]

# 🚀 定义多枚导弹的阵列参数 (名字必须与 UE4 世界大纲完全一致)
MISSILES_CONFIG = [
    # 导弹1：从正后方底层发射，飞向右上角
    {"name": "Missile_1", "start": [0, -15, -2], "end": [150, 20, -60]},
    # 导弹2：从正中心高层发射，直奔远方
    {"name": "Missile_2", "start": [0, 0, -10], "end": [150, 0, -50]},
    # 导弹3：从正后方右侧底层发射，飞向左上角
    {"name": "Missile_3", "start": [0, 15, -2], "end": [150, -20, -40]}
]

# 2. 摄像机固定机位 (将 X 设为 50，与 SHARED_TARGET_POS 的 X 坐标对齐)
# X=50: 实现正侧面视角
# Y=80: 将相机横向拉远（目标 Y 是 10，这里 Y=80 意味着距离 70 米），视野更开阔
# Z=-30: 将相机高度调至与目标点等高，获得平视效果
FIXED_CAMERA_POS = [50, 80, -30]


# =================================================

def main():
    print("🔌 正在连接 UE4...")
    client = airsim.VehicleClient()
    client.confirmConnection()

    # 1. 批量初始化所有的导弹控制器
    missile_controllers = []
    for cfg in MISSILES_CONFIG:
        ctrl = MissileController(
            client=client,
            missile_name=cfg["name"],
            start_pos=cfg["start"],
            target_pos=SHARED_TARGET_POS,
            end_pos=cfg["end"]
        )
        missile_controllers.append(ctrl)
        print(f"✅ 成功挂载导弹: {cfg['name']}")

    camera_rec = CameraRecorder(
        client=client,
        camera_name="0",
        save_dir=SAVE_DIR,
        record_rate=RECORD_RATE,
        camera_pos=FIXED_CAMERA_POS,
        look_at_pos=SHARED_TARGET_POS
    )

    final_video_path = ""

    try:
        # 2. 启动录像环境
        final_video_path = camera_rec.setup()
        print("🚀 蜂群导弹发射，多目标轨迹同步录制中...")

        start_time = time.time()

        for frame_id in range(TOTAL_FRAMES):
            loop_start = time.time()
            progress_t = frame_id / (TOTAL_FRAMES - 1)

            # 👉 核心变更：用字典收集所有导弹的状态，并提取数字 ID
            current_poses = {}
            for ctrl in missile_controllers:
                pose = ctrl.update_pose(progress_t)

                # 自动从 "Missile_1" 这种名字中提取数字 "1" 作为 MOT 标准的 track_id
                track_id = ctrl.missile_name.split('_')[-1]
                current_poses[track_id] = pose

            # 将包含了所有导弹位置和 ID 的字典一并传给相机
            current_time = time.time() - start_time
            camera_rec.record_frame(frame_id, current_time, current_poses)

            if frame_id % 15 == 0:
                print(f"▶️ 进度: {progress_t * 100:.1f}% | 帧: {frame_id}/{TOTAL_FRAMES} | 耗时: {current_time:.2f}s")

            elapsed = time.time() - loop_start
            sleep_time = RECORD_RATE - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("🎯 所有导弹抵达终点，任务完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户手动中止。")
    finally:
        if 'camera_rec' in locals():
            camera_rec.close()

        if final_video_path:
            print("-" * 50)
            print(f"🎬 多目标大片杀青！已保存至:\n👉 {final_video_path}")
            print("-" * 50)

        print("🔌 仿真环境已断开连接。")


if __name__ == "__main__":
    main()