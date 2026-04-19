import airsim
import time
import random
import argparse  # <--- 新增：用于解析命令行参数
import cv2  # <--- 新增：用于视频抽帧
import os
import re
from trajectory_control import MissileController
from camera_control import CameraRecorder

# ================= ⚙️ 蜂群(多导弹)全局配置 =================
SAVE_DIR = "H:/Missile_Video_Dataset"
PIC_SAVE_DIR = "H:/Missile_Picture_Dataset"  # <--- 新增：截图根目录

TOTAL_FRAMES = 180  # 总帧数 (录制 3 秒)
RECORD_RATE = 0.0167  # ≈ 60 FPS

# 🎯 核心交汇点
SHARED_TARGET_POS = [50, 10, -30]

# 🚀 定义多枚导弹的阵列参数
MISSILES_CONFIG = [
    {"name": "Missile_1", "start": [0, -15, -2], "end": [150, 20, -60]},
    {"name": "Missile_2", "start": [0, 0, -10], "end": [150, 0, -50]},
    {"name": "Missile_3", "start": [0, 15, -2], "end": [150, -20, -40]}
]


# =================================================

def extract_frames(video_path, frame_interval):
    """全自动抽帧引擎"""
    # 智能提取视频路径中的时间戳 (例如 20260414_114331)
    match = re.search(r"(\d{8}_\d{6})", video_path)
    time_str = match.group(1) if match else "UnknownTime"

    # 动态生成输出文件夹并创建
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

        # 满足切片参数 i 的间隔就截图
        if frame_count % frame_interval == 0:
            filename = f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"✅ 自动抽帧完成！共生成 {saved_count} 张高质量截图")
    print(f"📁 已安全保存至: {output_dir}\n")


def main():
    # ====== 🚀 新增：命令行参数解析 ======
    parser = argparse.ArgumentParser(description="AirSim 蜂群导弹仿真与全自动产线")
    # 切片参数 i：如果不输入 -i，默认为 0（表示不抽帧只录像）
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
        ctrl = MissileController(
            client=client,
            missile_name=cfg["name"],
            start_pos=cfg["start"],
            target_pos=SHARED_TARGET_POS,
            end_pos=cfg["end"]
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
            progress_t = frame_id / (TOTAL_FRAMES - 1)

            current_poses = {}
            for ctrl in missile_controllers:
                pose = ctrl.update_pose(progress_t)
                track_id = ctrl.missile_name.split('_')[-1]
                current_poses[track_id] = pose

            current_time = time.time() - start_time
            camera_rec.record_frame(frame_id, current_time, current_poses)

            if frame_id % 15 == 0:
                print(f"▶️ 进度: {progress_t * 100:.1f}% | 帧: {frame_id}/{TOTAL_FRAMES} | 耗时: {current_time:.2f}s",
                      end="\r")

            elapsed = time.time() - loop_start
            sleep_time = RECORD_RATE - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        print("\n🎯 所有导弹抵达终点，任务完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户手动中止。")
    finally:
        # 第一步：关闭并保存视频（必须先关闭，释放文件占用）
        if 'camera_rec' in locals():
            camera_rec.close()

        if final_video_path:
            print("-" * 50)
            print(f"🎬 多目标大片杀青！已保存至:\n👉 {final_video_path}")

            # 第二步：检测是否有切片参数，直接热乎地开始抽帧！
            if args.interval > 0:
                extract_frames(final_video_path, args.interval)
            print("-" * 50)

        print("🔌 仿真环境已断开连接。")


if __name__ == "__main__":
    main()