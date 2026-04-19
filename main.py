import airsim
import time
import random
import argparse
import cv2
import os
import re
from trajectory_control import MissileController
from camera_control import CameraRecorder

# ================= ⚙️ 蜂群(多导弹)全局配置 =================
SAVE_DIR = "H:/Missile_Video_Dataset"
PIC_SAVE_DIR = "H:/Missile_Picture_Dataset"

RECORD_RATE = 0.0167  # ≈ 60 FPS
# 因为最大延迟3秒 + 最大飞行时长4秒 = 最多飞7秒。7秒 * 60FPS = 420帧
TOTAL_FRAMES = 420

# 🎯 核心交汇点：现在仅作为摄像机“盯死”的目标点，导弹【不】会强行经过这里
SHARED_TARGET_POS = [50, 10, -30]

# 🚀 蜂群阵列：实现“盲盒式”错峰发射
# delay使用 0.0~3.0 之间的随机数，保证发射时间绝不超3秒！
MISSILES_CONFIG = [
    {"name": "Missile_1", "start": [0, -15, -2], "end": [150, 20, -60],
     "delay": random.uniform(0.0, 3.0), "duration": random.uniform(2.5, 4.0)},
    {"name": "Missile_2", "start": [0, 0, -10], "end": [150, 0, -50],
     "delay": random.uniform(0.0, 3.0), "duration": random.uniform(2.5, 4.0)},
    {"name": "Missile_3", "start": [0, 15, -2], "end": [150, -20, -40],
     "delay": random.uniform(0.0, 3.0), "duration": random.uniform(2.5, 4.0)}
]


# =================================================

def extract_frames(video_path, frame_interval):
    """全自动抽帧引擎"""
    match = re.search(r"(\d{8}_\d{6})", video_path)
    time_str = match.group(1) if match else "UnknownTime"

    output_dir = os.path.join(PIC_SAVE_DIR, f"Images_{time_str}")
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 错误: 无法读取刚生成的视频 {video_path}")
        return

    frame_count, saved_count = 0, 0
    print(f"\n🎞️ 正在从视频中以每 {frame_interval} 帧抽取截图...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ 自动抽帧完成！共生成 {saved_count} 张高质量截图")
    print(f"📁 已安全保存至: {output_dir}\n")


def main():
    parser = argparse.ArgumentParser(description="AirSim 蜂群导弹仿真与全自动产线")
    parser.add_argument("-i", "--interval", type=int, default=0, help="切片参数：每隔 i 帧自动截图 (默认 0: 不截图)")
    args = parser.parse_args()

    print("🔌 正在连接 UE4...")
    client = airsim.VehicleClient()
    client.confirmConnection()

    # ====== 随机“近战”固定机位 ======
    random_camera_pos = [
        random.uniform(20, 80),
        random.uniform(10, 30),
        random.uniform(-15, -2)
    ]
    print(
        f"🎲 本次运行生成的近战摄像机坐标: [X:{random_camera_pos[0]:.1f}, Y:{random_camera_pos[1]:.1f}, Z:{random_camera_pos[2]:.1f}]")

    missile_controllers = []
    for cfg in MISSILES_CONFIG:
        # ⚠️核心修复：去掉了旧版强制传入的 target_pos=SHARED_TARGET_POS，
        # 并正确传入了随机的 launch_delay 和 flight_duration
        ctrl = MissileController(
            client=client,
            missile_name=cfg["name"],
            start_pos=cfg["start"],
            end_pos=cfg["end"],
            launch_delay=cfg["delay"],
            flight_duration=cfg["duration"]
        )
        missile_controllers.append(ctrl)

    camera_rec = CameraRecorder(
        client=client,
        camera_name="0",
        save_dir=SAVE_DIR,
        record_rate=RECORD_RATE,
        camera_pos=random_camera_pos,
        look_at_pos=SHARED_TARGET_POS
    )

    final_video_path = ""

    try:
        final_video_path = camera_rec.setup()
        print("🚀 蜂群导弹发射，多目标轨迹同步录制中...")
        start_time = time.time()

        for frame_id in range(TOTAL_FRAMES):
            loop_start = time.time()

            # ⚠️核心修复：通过计算真实的“仿真世界秒数”来驱动导弹，从而使 delay 参数生效
            simulated_time = frame_id * RECORD_RATE

            current_poses = {}
            for ctrl in missile_controllers:
                # 传入秒数，导弹内部代码会自己判断是否过了延迟发射时间
                pose = ctrl.update_pose(simulated_time)
                track_id = ctrl.missile_name.split('_')[-1]
                current_poses[track_id] = pose

            actual_time_elapsed = time.time() - start_time
            camera_rec.record_frame(frame_id, actual_time_elapsed, current_poses)

            if frame_id % 15 == 0:
                progress = frame_id / TOTAL_FRAMES * 100
                print(f"▶️ 进度: {progress:.1f}% | 仿真时钟: {simulated_time:.2f}s | 帧: {frame_id}/{TOTAL_FRAMES}",
                      end="\r")

            elapsed = time.time() - loop_start
            sleep_time = RECORD_RATE - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\n🎯 所有导弹抵达终点，任务完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户手动中止。")
    finally:
        if 'camera_rec' in locals():
            camera_rec.close()

        if final_video_path:
            print("-" * 50)
            print(f"🎬 多目标大片杀青！已保存至:\n👉 {final_video_path}")

            if args.interval > 0:
                extract_frames(final_video_path, args.interval)
            print("-" * 50)

        print("🔌 仿真环境已断开连接。")


if __name__ == "__main__":
    main()