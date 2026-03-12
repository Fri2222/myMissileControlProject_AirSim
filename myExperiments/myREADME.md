####################################################################################################################

项目名称:myMissileControlProject_AirSim
环境名称:missile_sim


####################################################################################################################

1.pycharm打开工程文件myMissileControlProject_AirSim
	H:\Code\myMissileControlProject_AirSim
	
2.anaconda中切换环境为missile_sim
	conda activate missile_sim
	
运行
1.双击以下路径的脚本，目的是强制以纯净摄像机视角启动UE4工程
	"G:\WorkSpace\UnrealProjects\Missile_12_18\Start_Project.bat"

2.pycharm或anaconda中启动 main.py
	python main.py


python tools\train.py -f exps\default\yolox_missile.py -d 1 -b 8 --fp16 -c yolox_s.pth



主要提升召回率，更倾向于宁愿多预测一些错的也不能漏检



####################################################################################################################

├─main.py  //第二版main
│    ├─camera_control.py  //摄像头控制模块
│    │ 
│    └─trajectory_control.py  //轨迹控制模块
│
│
├─record_images_dataset.py  //将UE4模拟以图片格式保存，注意使用该文件需要使用第一版的main
├─record_trajectory.py  //注意使用该文件需要使用第一版的main
├─record_video_dataset.py  //将UE4模拟以视频格式保存,注意使用该文件需要使用第一版的main



####################################################################################################################
anaconda命令

1.conda info --envs

2.conda activate bytetrack

3.cd H:\Code\Byte\ByteTrack-main\ByteTrack-main






####################################################################################################################
git命令

1. 添加所有文件到暂存区
	git add .

2.提交文件到本地仓库
	git commit -m "这里写备注"
	
3.将本地代码推送到 GitHub
	git push -u origin <分支名>
	
4.查看当前分支
	git branch
	
5.创建分支
	git branch <分支名>
	
6.切换分支
	git checkout <分支名>