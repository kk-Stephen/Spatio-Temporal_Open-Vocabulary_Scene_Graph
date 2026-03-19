# -*- coding: utf-8 -*-
import os
import sys
import math
import time
from pathlib import Path

import cv2
import numpy as np
try:
    import pyrealsense2 as rs
except Exception as e:
    print("请先安装 Intel RealSense SDK 的 Python 包： pip install pyrealsense2")
    raise

# ========= 在这里修改输入/输出路径 =========
INPUT_BAG  = r"H:\DATA\dataset_planning\depth\depth\depth_57.bag"
OUTPUT_DIR = r"C:\latencygraph\testdata\paperpic"  # 将在此目录下生成 0_rgb.png/0_depth.png ...
FPS_TARGET = 2  # 目标抽帧帧率
# =========================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def preprocess(input_path, output_dir, fps):
    bag_path = Path(input_path)
    out_dir  = Path(output_dir)

    if not bag_path.exists():
        print(f"[错误] 找不到文件：{bag_path}")
        sys.exit(1)

    ensure_dir(out_dir)
    print(f"[信息] 输出目录：{out_dir.resolve()}")

    # 配置 RealSense 管线以播放 .bag
    pipeline = rs.pipeline()
    config   = rs.config()

    # 从文件读取，不循环播放
    config.enable_device_from_file(str(bag_path), repeat_playback=False)

    # 不强制指定分辨率/格式，沿用 .bag 文件内的流设置
    # 如需手动对齐深度到彩色
    align_to_color = rs.align(rs.stream.color)

    # 启动
    profile = pipeline.start(config)

    # 让播放以“非实时”模式跑，避免因系统速度导致跳帧
    playback = profile.get_device().as_playback()
    playback.set_real_time(False)

    # 查询 depth scale（将原始单位换算为米）
    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale  = depth_sensor.get_depth_scale()  # meters per unit

    # 深度后处理滤波器（降低散斑与空洞）
    spatial  = rs.spatial_filter()   # 空间滤波
    temporal = rs.temporal_filter()  # 时间滤波
    hole     = rs.hole_filling_filter()  # 填洞

    # 目标抽帧间隔（毫秒）
    save_period_ms = 1000.0 / float(fps)
    last_save_ts   = None  # 毫秒时间戳（使用 color 帧）

    saved_count = 0

    print(f"[信息] 开始读取 .bag（目标 {fps} FPS 抽帧）...")
    try:
        while True:
            # 到文件末尾会抛 RuntimeError
            frames = pipeline.wait_for_frames()

            # 对齐：将深度对齐到彩色坐标
            frames = align_to_color.process(frames)

            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            # 使用彩色帧时间戳来抽帧（单位毫秒）
            ts_ms = color_frame.get_timestamp()

            if (last_save_ts is None) or (ts_ms - last_save_ts >= save_period_ms):
                last_save_ts = ts_ms

                # ---------- 获取 Numpy ----------
                color_np = np.asanyarray(color_frame.get_data())  # HxWx3, BGR? RealSense 默认是RGB格式
                # RealSense Python SDK拿到的是RGB顺序，这里统一转为 BGR 以便用 OpenCV 保存
                color_np_bgr = cv2.cvtColor(color_np, cv2.COLOR_RGB2BGR)

                # 深度滤波（RealSense 空间/时间/填洞）
                df = depth_frame
                df = spatial.process(df)
                df = temporal.process(df)
                df = hole.process(df)

                depth_np_raw = np.asanyarray(df.get_data()).astype(np.uint16)  # 原始单位（uint16）

                # 将深度转换为毫米（便于可视化/使用；保留 uint16）
                # meters = raw * depth_scale
                # millimeters = meters * 1000
                depth_mm = np.rint(depth_np_raw.astype(np.float32) * depth_scale * 1000.0).astype(np.uint16)

                # 额外的平滑：RGB 用双边滤波，Depth 用中值滤波（不会改变位深）
                # 双边滤波保边缘，去噪（参数可按需调整）
                color_smooth = cv2.bilateralFilter(color_np_bgr, d=7, sigmaColor=75, sigmaSpace=75)
                # color_smooth = cv2.bilateralFilter(color_np, d=7, sigmaColor=75, sigmaSpace=75)
                # 深度中值滤波（对 16U 可用）
                depth_smooth = cv2.medianBlur(depth_mm, ksize=5)

                # ---------- 保存 ----------
                rgb_name   = out_dir / f"{saved_count}_rgb.png"
                depth_name = out_dir / f"{saved_count}_depth.png"

                # 保存 RGB（8-bit 3通道）
                ok_rgb = cv2.imwrite(str(rgb_name), color_smooth)
                # 保存 Depth（16-bit 单通道，单位：毫米）
                ok_dep = cv2.imwrite(str(depth_name), depth_smooth)

                if not ok_rgb or not ok_dep:
                    print(f"[警告] 第 {saved_count} 帧保存失败：{rgb_name} / {depth_name}")
                else:
                    if saved_count % 20 == 0:
                        print(f"[进度] 已保存 {saved_count} 对帧")

                saved_count += 1

    except RuntimeError:
        # 读到文件末尾
        pass
    finally:
        pipeline.stop()

    print(f"[完成] 共保存 {saved_count} 对 RGB/Depth 帧到：{out_dir.resolve()}")
    print("[说明] 深度图为 16-bit PNG，数值单位为毫米（0 表示无效/缺失测量）。")

if __name__ == "__main__":
    preprocess(INPUT_BAG, OUTPUT_DIR, FPS_TARGET)