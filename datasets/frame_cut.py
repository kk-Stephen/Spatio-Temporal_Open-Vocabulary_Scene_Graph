import cv2
import os
import sys

# 要用绝对路径：
input_mp4 = r"C:\latencygraph\testdata\test_rgbd_videos\output_videos\depth_20250806-211909.avi"
output_dir = r"C:\latencygraph\testdata\test_rgbd_videos\test_video_d_frames"

# 打印当前工作目录，帮助定位相对路径问题
print("current path：", os.getcwd())

# 检查视频文件是否存在
if not os.path.exists(input_mp4):
    raise IOError(f"No video path: {input_mp4}")

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 打开视频
cap = cv2.VideoCapture(input_mp4)
if not cap.isOpened():
    raise IOError(f"Cannot open video: {input_mp4}")

# 获取帧率并计算保存间隔
fps = cap.get(cv2.CAP_PROP_FPS)
if fps <= 0:
    raise ValueError("Cannot get fps，fps =", fps)

# 0.5 秒 / 帧 => 每隔 fps * 0.5 帧保存一次
interval = max(1, int(round(fps * 0.2)))

frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval == 0:
        out_path = os.path.join(output_dir, f"frame_{saved_idx:04d}.png")
        cv2.imwrite(out_path, frame)
        saved_idx += 1

    frame_idx += 1

cap.release()
print(f"Complete saving {saved_idx}  frame to {output_dir}")
