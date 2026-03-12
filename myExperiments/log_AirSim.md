# AIRSIM 实验记录日志

| 实验ID | 日期 | Commit Hash | 关键改动 (Parameters) | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **Exp-001** | 2026-03-08 | `单导弹 移动镜头视频 |
| **Exp-002** | 2026-03-12 | `Exp-002 固定镜头视频，模块解耦，main分别调用轨迹控制和摄像头控制 | 单导弹 | 
| **Exp-003** | 2026-03-12 | `Exp-003 随机轨迹但轨迹过固定点 | 单导弹 | 



###################################################################################################################


## 详细数据备份
### Exp-001 
- **Code Version**: `单导弹 移动镜头视频`
- **Command**: `python tools/track.py -f exps/example/mot/yolox_s_mot17_half.py -c pretrained/bytetrack_s_mot17.pth.tar -b 1 -d 1 --fp16 --fuse`
- **Output**:
			"H:\Missile_Video_Dataset\VideoData_FixedView_20251226_101912"