# compress_videos.py

import os
import subprocess
import argparse
import json
import shutil  # [新增] 导入shutil库用于文件复制
from tqdm import tqdm


def get_video_duration(video_path: str) -> float:
    """使用 ffprobe 获取视频的时长（秒）。"""
    command = [
        'ffprobe',
        '-v', 'error',
        '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    try:
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
        return float(result.stdout)
    except Exception as e:
        print(f"错误：无法获取视频 '{video_path}' 的时长。请确保 ffprobe 已正确安装。错误: {e}")
        return 0.0


def compress_video(input_path: str, output_path: str, target_size_mb: float):
    """
    压缩单个视频文件，使其分辨率降低且文件大小接近目标值。
    """
    duration = get_video_duration(input_path)
    if duration <= 0:
        print(f"跳过压缩 {input_path}，因为无法获取其时长。")
        return False

    target_total_bitrate_kbps = (target_size_mb * 8192) / duration
    audio_bitrate_kbps = 128
    video_bitrate_kbps = target_total_bitrate_kbps - audio_bitrate_kbps

    if video_bitrate_kbps <= 0:
        print(f"警告：为 '{os.path.basename(input_path)}' 计算出的视频比特率过低。将使用最低值。")
        video_bitrate_kbps = 100

    command = [
        'ffmpeg',
        '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-b:v', f'{int(video_bitrate_kbps)}k',
        '-vf', 'scale=854:-2',
        '-c:a', 'aac',
        '-b:a', f'{audio_bitrate_kbps}k',
        '-loglevel', 'error',  # [修改] 只在发生真正错误时才打印ffmpeg日志
        output_path
    ]

    try:
        subprocess.run(command, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n处理文件 '{input_path}' 时 ffmpeg 发生错误。")
        return False
    except FileNotFoundError:
        print("错误: 'ffmpeg' 命令未找到。请确保 FFmpeg 已正确安装并添加到了系统环境变量 Path 中。")
        exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="批量处理文件夹中的 .mp4 文件。只压缩超过特定大小的视频，其余则直接复制。")
    parser.add_argument("input_dir", type=str, help="包含源视频文件的文件夹路径。")
    parser.add_argument("output_dir", type=str, help="用于保存处理后视频的文件夹路径。")
    parser.add_argument("--target_size", type=float, default=8.0, help="压缩视频的目标最大文件大小（MB）。")
    # [新增] 添加一个新的命令行参数用于设置大小阈值
    parser.add_argument("--size_threshold", type=float, default=8.0,
                        help="文件大小阈值（MB）。只有大于此大小的文件才会被压缩。")

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    video_files = [f for f in os.listdir(args.input_dir) if f.lower().endswith('.mp4')]
    if not video_files:
        print(f"在文件夹 '{args.input_dir}' 中没有找到 .mp4 文件。")
        return

    print(f"找到 {len(video_files)} 个 .mp4 文件。开始处理...")
    print(f"将压缩所有大于 {args.size_threshold} MB 的文件，目标大小为 {args.target_size} MB。")

    # [修改] 更新循环以处理条件逻辑
    for filename in tqdm(video_files, desc="视频处理进度"):
        input_path = os.path.join(args.input_dir, filename)
        output_path = os.path.join(args.output_dir, filename)

        # 步骤 1: 检查文件大小
        try:
            file_size_bytes = os.path.getsize(input_path)
            file_size_mb = file_size_bytes / (1024 * 1024)
        except FileNotFoundError:
            print(f"警告：文件 '{input_path}' 不存在，跳过。")
            continue

        # 步骤 2: 条件判断
        if file_size_mb > args.size_threshold:
            # 文件太大，需要压缩
            tqdm.write(f"压缩中: {filename} ({file_size_mb:.2f} MB > {args.size_threshold} MB)")
            compress_video(input_path, output_path, args.target_size)
        else:
            # 文件大小合适，直接复制
            tqdm.write(f"复制中: {filename} ({file_size_mb:.2f} MB <= {args.size_threshold} MB)")
            shutil.copy2(input_path, output_path)

    print("\n所有视频处理完成！")
    print(f"处理后的文件保存在: {args.output_dir}")


if __name__ == "__main__":
    main()