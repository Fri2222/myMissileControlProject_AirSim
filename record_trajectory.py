import airsim
import time
import os
import csv
import datetime

# ================= 配置区域 =================
MISSILE_NAME = "Missile_1"  # UE4 中导弹的名字
RECORD_RATE = 0.05  # 录制间隔 (秒)，0.05 = 20Hz (每秒20次)
SAVE_DIR = "G:/Missile_Data"  # 数据保存路径 (建议改成你的实际路径)


# ===========================================

def setup_writer():
    """初始化保存目录和 CSV 文件"""
    # 1. 创建带时间戳的文件夹，防止覆盖旧数据
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    current_dir = os.path.join(SAVE_DIR, f"Flight_{timestamp}")

    if not os.path.exists(current_dir):
        os.makedirs(current_dir)
        print(f"📂 数据集目录已创建: {current_dir}")

    # 2. 创建 CSV 文件
    csv_path = os.path.join(current_dir, "trajectory_ground_truth.csv")
    csv_file = open(csv_path, 'w', newline='')
    writer = csv.writer(csv_file)

    # 3. 写入表头 (Header)
    # Timestamp: 时间戳
    # POS: 位置 (NED坐标系, 米)
    # ORT: 姿态 (四元数 W, X, Y, Z)
    header = [
        "timestamp",
        "pos_x", "pos_y", "pos_z",
        "ort_w", "ort_x", "ort_y", "ort_z"
    ]
    writer.writerow(header)

    return csv_file, writer, csv_path


def main():
    # 连接 AirSim
    client = airsim.VehicleClient()
    try:
        client.confirmConnection()
        print(f"✅ 已连接到 AirSim，准备录制: {MISSILE_NAME}")
    except Exception as e:
        print(f"❌ 连接失败: {e}\n👉 请确保 UE4 正在运行并且点击了 '播放 (Play)'！")
        return

    # 初始化文件写入
    try:
        f, writer, path = setup_writer()
    except Exception as e:
        print(f"❌ 无法创建文件: {e}")
        return

    print(f"🔴 开始录制... (按 Ctrl+C 停止保存)")
    print(f"   数据将保存至: {path}")

    start_time = time.time()

    try:
        while True:
            # 1. 获取导弹当前的位姿 (Ground Truth)
            # 这是一个阻塞调用，确保拿到的那一刻是最新的
            pose = client.simGetObjectPose(MISSILE_NAME)

            # 检查是否真的获取到了数据 (防止空数据)
            if pose.position.x_val == 0 and pose.position.y_val == 0 and pose.position.z_val == 0:
                # 有时候没获取到会返回全0，可以选择跳过或记录
                pass

                # 2. 计算相对时间
            current_time = time.time() - start_time

            # 3. 提取数据
            pos = pose.position
            ort = pose.orientation

            row = [
                f"{current_time:.4f}",  # 时间戳 (保留4位小数)
                f"{pos.x_val:.6f}",  # X
                f"{pos.y_val:.6f}",  # Y
                f"{pos.z_val:.6f}",  # Z
                f"{ort.w_val:.6f}",  # 四元数 W
                f"{ort.x_val:.6f}",  # 四元数 X
                f"{ort.y_val:.6f}",  # 四元数 Y
                f"{ort.z_val:.6f}"  # 四元数 Z
            ]

            # 4. 写入文件
            writer.writerow(row)

            # 5. 控制台打印状态 (每10次打印一次，避免刷屏太快)
            # end='\r' 让光标回到行首，实现动态刷新效果
            print(f"📍 正在录制 [T={current_time:.1f}s]: X={pos.x_val:.2f}, Y={pos.y_val:.2f}, Z={pos.z_val:.2f}",
                  end='\r')

            # 6. 等待下一帧
            time.sleep(RECORD_RATE)

    except KeyboardInterrupt:
        print("\n\n🛑 录制手动停止。")
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
    finally:
        f.close()
        print("💾 文件已安全保存并关闭。")


if __name__ == "__main__":
    main()