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

        # ====== 新增：独立的截屏存储配置 ======
        self.pic_save_base_dir = r"H:\Missile_Picture_Dataset"
        self.capture_interval = 30  # 每隔 30 帧截屏一次
        self.current_pic_dir = None
        # ======================================

    def _calculate_look_at_quaternion(self):
        dx = self.look_at_pos.x_val - self.camera_pos.x_val
        dy = self.look_at_pos.y_val - self.camera_pos.y_val
        dz = self.look_at_pos.z_val - self.camera_pos.z_val

        yaw = math.atan2(dy, dx)
        distance_xy = math.sqrt(dx * dx + dy * dy)
        pitch = -math.atan2(dz, distance_xy)
        return airsim.to_quaternion(pitch, 0, yaw)

    def setup(self):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # 原有功能：配置视频和 CSV 存储目录
        current_dir = os.path.join(self.save_dir, f"Dataset_FixedView_{timestamp}")
        os.makedirs(current_dir, exist_ok=True)

        self.csv_file = open(os.path.join(current_dir, "ground_truth.csv"), 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        self.writer.writerow(
            ["frame_id", "timestamp", "track_id", "pos_x", "pos_y", "pos_z", "ort_w", "ort_x", "ort_y", "ort_z"])

        # ====== 新增：配置专属的截屏子目录 ======
        self.current_pic_dir = os.path.join(self.pic_save_base_dir, f"Images_{timestamp}")
        os.makedirs(self.current_pic_dir, exist_ok=True)
        # ========================================

        fixed_rotation = self._calculate_look_at_quaternion()
        self.client.simSetVehiclePose(airsim.Pose(self.camera_pos, fixed_rotation), True)

        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)])
        img_width, img_height = responses[0].width, responses[0].height

        video_path = os.path.join(current_dir, "flight_video.avi")
        self.video_writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'XVID'), self.fps,
                                            (img_width, img_height))

        print(f"📷 录像已准备就绪: {img_width}x{img_height} @ {self.fps}FPS")
        print(f"📸 独立截屏功能已开启: 每 {self.capture_interval} 帧抓拍，保存至 {self.current_pic_dir}")
        return video_path

    def record_frame(self, frame_id, current_time, poses_dict):
        """
        抓取一帧并记录。
        poses_dict: 字典格式，例如 { "1": pose1, "2": pose2 }
        """
        responses = self.client.simGetImages(
            [airsim.ImageRequest(self.camera_name, airsim.ImageType.Scene, False, False)])
        if responses[0].width == 0:
            return

        img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
        try:
            img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
        except ValueError:
            img_bgr = img1d.reshape(responses[0].height, responses[0].width, 4)[:, :, :3]
        img_bgr = np.ascontiguousarray(img_bgr)

        # 视频只存一次 (原功能完整保留)
        self.video_writer.write(img_bgr)

        # ====== 新增：每 30 帧截屏并保存图片 ======
        if frame_id % self.capture_interval == 0:
            img_filename = f"frame_{frame_id:06d}.jpg"
            img_filepath = os.path.join(self.current_pic_dir, img_filename)
            cv2.imwrite(img_filepath, img_bgr)
        # ==========================================

        # 遍历所有导弹，在同一帧下写入多行数据 (原功能完整保留)
        for track_id, missile_pose in poses_dict.items():
            pos, ort = missile_pose.position, missile_pose.orientation
            self.writer.writerow([
                frame_id, f"{current_time:.4f}", track_id,
                f"{pos.x_val:.6f}", f"{pos.y_val:.6f}", f"{pos.z_val:.6f}",
                f"{ort.w_val:.6f}", f"{ort.x_val:.6f}", f"{ort.y_val:.6f}", f"{ort.z_val:.6f}"
            ])

    def close(self):
        if self.csv_file: self.csv_file.close()
        if self.video_writer: self.video_writer.release()
        print("💾 录制模块已安全关闭，文件已保存。")