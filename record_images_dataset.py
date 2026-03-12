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
RECORD_RATE = 0.0167  # 0.0167s ≈ 60 FPS
SAVE_DIR = "H:/Missile_Video_Dataset"
CAMERA_NAME = "0"

# 🎥 固定机位配置
FIXED_CAMERA_POS = airsim.Vector3r(50.0, 30.0, 0.0)


# ===============================================

def setup_dataset(client):
    """初始化目录、CSV和图片文件夹"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # 为了区分，文件夹名字改叫 ImageData
    current_dir = os.path.join(SAVE_DIR, f"ImageData_FixedView_{timestamp}")
    os.makedirs(current_dir, exist_ok=True)

    # 🌟 核心修改：创建专门存放照片的 images 文件夹
    images_dir = os.path.join(current_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    print(f"📂 数据集目录已创建: {current_dir}")

    csv_path = os.path.join(current_dir, "ground_truth.csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)
    header = ["frame_id", "timestamp", "pos_x", "pos_y", "pos_z", "ort_w", "ort_x", "ort_y", "ort_z"]
    writer.writerow(header)

    return csv_file, writer, current_dir, images_dir


def process_airsim_image(response):
    """处理 AirSim 图像，确保内存连续性"""
    img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
    try:
        img_bgr = img1d.reshape(response.height, response.width, 3)
    except ValueError:
        img_bgra = img1d.reshape(response.height, response.width, 4)
        img_bgr = img_bgra[:, :, :3].copy()
    return np.ascontiguousarray(img_bgr)


def calculate_look_at_quaternion(cam_pos, target_pos):
    """计算相机盯着导弹所需的旋转四元数"""
    dx = target_pos.x_val - cam_pos.x_val
    dy = target_pos.y_val - cam_pos.y_val
    dz = target_pos.z_val - cam_pos.z_val
    yaw = math.atan2(dy, dx)
    distance_xy = math.sqrt(dx * dx + dy * dy)
    pitch = -math.atan2(dz, distance_xy)
    return airsim.to_quaternion(pitch, 0, yaw)


def main():
    client = airsim.VehicleClient()
    try:
        client.confirmConnection()
        print(f"✅ AirSim 连接成功 | 模式：图片序列保存")
    except Exception as e:
        print(f"❌ 连接失败: {e}")
        return

    csv_f, writer, path, images_dir = None, None, None, None
    try:
        csv_f, writer, path, images_dir = setup_dataset(client)
    except Exception as e:
        print(f"❌ 初始化失败: {e}")
        return

    print(f"🔴 录制开始... (按 Ctrl+C 停止)")
    start_time = time.time()
    frame_id = 0

    try:
        while True:
            loop_start = time.time()

            # 1. 获取导弹真值并控制相机视角
            pose = client.simGetObjectPose(MISSILE_NAME)
            target_look_rotation = calculate_look_at_quaternion(FIXED_CAMERA_POS, pose.position)
            client.simSetVehiclePose(airsim.Pose(FIXED_CAMERA_POS, target_look_rotation), True)

            # 2. 拍照
            responses = client.simGetImages([airsim.ImageRequest(CAMERA_NAME, airsim.ImageType.Scene, False, False)])
            current_timestamp = time.time() - start_time
            img_response = responses[0]

            if img_response.width == 0: continue
            frame_bgr = process_airsim_image(img_response)

            # 3. 🌟 核心修改：直接保存为清晰的 JPG 图片 (六位数字补零格式，如 000005.jpg)
            img_path = os.path.join(images_dir, f"{frame_id:06d}.jpg")
            cv2.imwrite(img_path, frame_bgr)

            # 4. 写入 CSV 真值
            pos = pose.position
            ort = pose.orientation
            row = [
                frame_id, f"{current_timestamp:.4f}",
                f"{pos.x_val:.6f}", f"{pos.y_val:.6f}", f"{pos.z_val:.6f}",
                f"{ort.w_val:.6f}", f"{ort.x_val:.6f}", f"{ort.y_val:.6f}", f"{ort.z_val:.6f}"
            ]
            writer.writerow(row)

            if frame_id % 10 == 0:
                print(
                    f"📸 已存图片: {frame_id} | FPS: {1 / (time.time() - loop_start):.1f} | T: {current_timestamp:.2f}s",
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
        try:
            client.reset()
            client.enableApiControl(False)
        except:
            pass
        print(f"💾 图片序列保存完毕: {path}")


if __name__ == "__main__":
    main()