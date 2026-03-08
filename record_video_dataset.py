import airsim
import time
import os
import csv
import datetime
import cv2
import numpy as np
import math

# ================= ⚙️ 配置区域 =================
MISSILE_NAME = "Missile_1"
RECORD_RATE = 0.0167  # 0.0167s ≈ 60 FPS (提高帧率)
SAVE_DIR = "H:/Missile_Video_Dataset"
CAMERA_NAME = "0"

# --- 🎥 固定机位配置 ---
# 我们把相机放在飞行路径的中段侧面，这样能拍到全过程
# 假设导弹从 (0,0) 飞到 (50,0)，我们将相机放在 x=25 的地方
# Y=20 表示在右侧20米，Z=-5 表示高度5米
# NED坐标系：X前，Y右，Z下 (负数是天上)
# 假设导弹从 (0,0,0) 飞向 (100,0,0)

# 🎥 视角 A：完全侧面平拍 (类似横版过关游戏)
# 在 x=50处(路程中间), y=30(右侧30米), z=0(和导弹同高)
FIXED_CAMERA_POS = airsim.Vector3r(50.0, 30.0, 0.0)




# ===============================================

def setup_dataset(client):
    """初始化目录、CSV和视频"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.join(SAVE_DIR, f"VideoData_FixedView_{timestamp}")
    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
        print(f"📂 数据集目录已创建: {current_dir}")

    csv_path = os.path.join(current_dir, "ground_truth.csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = ["frame_id", "timestamp", "pos_x", "pos_y", "pos_z", "ort_w", "ort_x", "ort_y", "ort_z"]
    writer.writerow(header)

    print("正在获取相机分辨率...")
    responses = client.simGetImages([airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)])
    if not responses or responses[0].width == 0:
        raise Exception("无法获取图像！请检查UE4。")

    img_width = responses[0].width
    img_height = responses[0].height
    print(f"📷 录制分辨率: {img_width} x {img_height} @ {int(1 / RECORD_RATE)} FPS")

    video_path = os.path.join(current_dir, "flight_video.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = int(1 / RECORD_RATE)
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (img_width, img_height))

    return csv_file, writer, video_writer, current_dir


def process_airsim_image(response):
    """
    AirSim 返回的 uncompressed 图片通常已经是 BGR 格式了 (或者 BGRA)
    """
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)

    # 1. 尝试重塑为 H x W x 3 (BGR)
    try:
        img_bgr = img1d.reshape(response.height, response.width, 3)
    except ValueError:
        # 如果报错，说明可能是 BGRA (4通道)，需要切掉 Alpha 通道
        img_bgra = img1d.reshape(response.height, response.width, 4)
        img_bgr = img_bgra[:, :, :3]

    # 2. 直接返回，不要做 RGB2BGR 转换
    return img_bgr

def calculate_look_at_quaternion(cam_pos, target_pos):
    """
    数学核心：计算让 A 点看向 B 点所需的旋转四元数 (Pan/Tilt)
    """
    # 1. 计算相对向量 (从相机指向目标)
    dx = target_pos.x_val - cam_pos.x_val
    dy = target_pos.y_val - cam_pos.y_val
    dz = target_pos.z_val - cam_pos.z_val

    # 2. 计算偏航角 (Yaw) - 水平旋转
    # atan2(y, x) 返回弧度
    yaw = math.atan2(dy, dx)

    # 3. 计算俯仰角 (Pitch) - 垂直旋转
    # 计算水平距离
    distance_xy = math.sqrt(dx * dx + dy * dy)
    # atan2(z, dist)
    # 注意：AirSim中 Z 轴向下是正。如果目标在相机上方，dz是负数，pitch应该是负数(抬头)。
    # math.atan2 能够正确处理这个符号。
    pitch = math.atan2(dz, distance_xy)
    pitch = -pitch
    # 4. 生成四元数 (Roll 保持为 0)
    return airsim.to_quaternion(pitch, 0, yaw)


def main():
    client = airsim.VehicleClient()
    try:
        client.confirmConnection()
        print(f"✅ AirSim 连接成功 | 模式：固定机位跟踪 | 帧率：60 FPS")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    csv_f, writer, v_writer, path = None, None, None, None
    try:
        csv_f, writer, v_writer, path = setup_dataset(client)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print(f"🔴 录制开始... (按 Ctrl+C 停止)")

    start_time = time.time()
    frame_id = 0

    try:
        while True:
            loop_start = time.time()

            # 1. 获取导弹真值
            pose = client.simGetObjectPose(MISSILE_NAME)

            # 2. 【关键修改】相机位置固定，只计算旋转
            # 计算相机需要转多少度才能盯着导弹
            target_look_rotation = calculate_look_at_quaternion(FIXED_CAMERA_POS, pose.position)

            # 更新 AirSim 观察者位姿 (位置不变，旋转变化)
            client.simSetVehiclePose(airsim.Pose(FIXED_CAMERA_POS, target_look_rotation), True)

            # 3. 拍照
            responses = client.simGetImages([airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)])
            current_timestamp = time.time() - start_time
            img_response = responses[0]

            if img_response.width == 0: continue

            # 4. 写入数据
            frame_bgr = process_airsim_image(img_response)
            v_writer.write(frame_bgr)

            pos = pose.position
            ort = pose.orientation
            row = [
                frame_id, f"{current_timestamp:.4f}",
                f"{pos.x_val:.6f}", f"{pos.y_val:.6f}", f"{pos.z_val:.6f}",
                f"{ort.w_val:.6f}", f"{ort.x_val:.6f}", f"{ort.y_val:.6f}", f"{ort.z_val:.6f}"
            ]
            writer.writerow(row)

            # 每10帧刷新一次打印，减少IO消耗提高性能
            if frame_id % 10 == 0:
                print(f"🎥 Frame: {frame_id} | FPS: {1 / (time.time() - loop_start):.1f} | T: {current_timestamp:.2f}s",
                      end='\r')

            frame_id += 1

            elapsed = time.time() - loop_start
            sleep_time = RECORD_RATE - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n\n🛑 录制停止。")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
    finally:
        if csv_f: csv_f.close()
        if v_writer: v_writer.release()
        print(f"💾 视频保存完毕: {path}")


if __name__ == "__main__":
    main()