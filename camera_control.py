import airsim
import os
import csv
import datetime
import cv2
import numpy as np
import math


class CameraRecorder:
    def __init__(self, client, camera_name, save_dir, record_rate, camera_pos, look_at_pos):
        self.client = client
        self.camera_name = str(camera_name)
        self.save_dir = save_dir
        self.record_rate = record_rate
        self.fps = int(1 / record_rate)

        self.camera_pos = airsim.Vector3r(*camera_pos)
        self.look_at_pos = airsim.Vector3r(*look_at_pos)

        self.csv_file = None
        self.writer = None
        self.video_writer = None

    def _calculate_look_at_quaternion(self):
        """计算让相机看向目标点的四元数"""
        dx = self.look_at_pos.x_val - self.camera_pos.x_val
        dy = self.look_at_pos.y_val - self.camera_pos.y_val
        dz = self.look_at_pos.z_val - self.camera_pos.z_val

        yaw = math.atan2(dy, dx)
        distance_xy = math.sqrt(dx * dx + dy * dy)
        pitch = -math.atan2(dz, distance_xy)
        return airsim.to_quaternion(pitch, 0, yaw)

    def setup(self):
        """初始化文件夹、锁定机位、创建视频流"""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        current_dir = os.path.join(self.save_dir, f"Dataset_FixedView_{timestamp}")
        os.makedirs(current_dir, exist_ok=True)

        # 初始化 CSV
        self.csv_file = open(os.path.join(current_dir, "ground_truth.csv"), 'w', newline='')
        self.writer = csv.writer(self.csv_file)
        self.writer.writerow(["frame_id", "timestamp", "pos_x", "pos_y", "pos_z", "ort_w", "ort_x", "ort_y", "ort_z"])

        # 锁定摄像机机位
        fixed_rotation = self._calculate_look_at_quaternion()
        self.client.simSetVehiclePose(airsim.Pose(self.camera_pos, fixed_rotation), True)

        # 初始化视频流
        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)])
        img_width, img_height = responses[0].width, responses[0].height

        video_path = os.path.join(current_dir, "flight_video.avi")
        # 使用 XVID 编码确保 Windows 下必定保存成功
        self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps,
                                            (img_width, img_height))

        print(f"📷 录像已准备就绪: {img_width}x{img_height} @ {self.fps}FPS -> {video_path}")

        return video_path
    def record_frame(self, frame_id, current_time, missile_pose):
        """抓取一帧并记录"""
        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)])
        if responses[0].width == 0:
            return

        # 处理图片内存连续性
        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        try:
            img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
        except ValueError:
            img_bgr = img1d.reshape(responses[0].height, responses[0].width, 4)[:, :, :3]
        img_bgr = np.ascontiguousarray(img_bgr)

        # 写入视频
        self.video_writer.write(img_bgr)

        # 写入 CSV 真值
        pos, ort = missile_pose.position, missile_pose.orientation
        self.writer.writerow([
            frame_id, f"{current_time:.4f}",
            f"{pos.x_val:.6f}", f"{pos.y_val:.6f}", f"{pos.z_val:.6f}",
            f"{ort.w_val:.6f}", f"{ort.x_val:.6f}", f"{ort.y_val:.6f}", f"{ort.z_val:.6f}"
        ])

    def close(self):
        """清理资源"""
        if self.csv_file: self.csv_file.close()
        if self.video_writer: self.video_writer.release()
        print("💾 录制模块已安全关闭，文件已保存。")