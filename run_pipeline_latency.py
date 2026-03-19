# main.py
from tqdm import tqdm
import sys
import os
import torch
import pandas as pd
import argparse
import json
from util.utils import get_model, build_cfg_from_yaml, get_rgb_depth_paths
from scene_graph import create_video_scene_graph
from planner import generate_prompt
# from preprocessing import preprocess # 假设预处理已完成，注释掉此行
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(description="Batch process videos to generate prompts for robot planning.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct",
                        help="LVLM model for scene description.")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run models on ('cuda' or 'cpu').")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to the main directory containing video subfolders (e.g., video_0, video_1).")
    parser.add_argument("--output_prompt_dir", type=str, default="full_prompts",
                        help="Directory to save the generated full_prompt .txt files.")
    parser.add_argument("--output_csv_path", type=str, default="planning_dataset.csv",
                        help="Path to save the final output CSV file.")
    parser.add_argument("--csv_path", type=str, default="planning_dataset.csv",)
    parser.add_argument("--fps", type=int, default=2, help="Frame rate of the videos.")
    args = parser.parse_args()

    # --- 1. 初始化 ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 创建输出文件夹
    os.makedirs(args.output_prompt_dir, exist_ok=True)

    # --- 2. 一次性加载所有大模型 ---
    print("Loading models... (This may take a moment)")
    # 加载用于场景描述的LVLM
    LVLM, _ = get_model(args, device)

    # 加载 Grounded DINO
    sys.path.insert(0, os.path.abspath("Grounded_SAM_2_main"))
    dino_cfg = build_cfg_from_yaml("configs/Grounded_DINO.yaml")
    Ground_DINO, _ = get_model(dino_cfg, device)
    print("Models loaded successfully.")

    df = pd.read_csv(args.csv_path)

    # --- 3. 初始化CSV数据收集器 ---
    results_data = []


    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating plans"):
        print("\n" + "=" * 50)
        video_name = row['video_name']
        instruction = row['instruction']
        com_ts = row['com_ts']
        latency = row['latency']
        print(f"Processing Video: {video_name}")
        print("=" * 50)

        current_video_dir = os.path.join(args.video_dir, video_name)

        # --- 4a. 为当前视频构建场景图 (每个视频只构建一次) ---
        print(f"Step 1: Building Video Scene Graph for {video_name}...")
        rgb_paths, depth_paths = get_rgb_depth_paths(directory=current_video_dir)

        if not rgb_paths or not depth_paths:
            print(f"Warning: No RGB or Depth frames found in {current_video_dir}. Skipping.")
            continue

        # 相机和深度参数 (可以根据需要移到 argparse 中)
        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3)
        depth_scale = 5000.0
        depth_trunc = 5.0

        video_sg_builder = create_video_scene_graph(
            LVLM=LVLM,
            Ground_DINO=Ground_DINO,
            frame_dir=current_video_dir,
            rgb_paths=rgb_paths,
            depth_paths=depth_paths,
            camera_intrinsics=camera_intrinsics,
            depth_scale=depth_scale,
            depth_trunc=depth_trunc,
            latency=latency
        )
        print("Scene Graph built successfully.")

        full_prompt = generate_prompt(video_sg_builder, user_prompt=str(instruction), com_ts=com_ts)


        prompt_filename = f"{video_name}_latency.txt"
        prompt_filepath = os.path.join(args.output_prompt_dir, prompt_filename)

        with open(prompt_filepath, "w", encoding="utf-8") as f:
                f.write(full_prompt)


        # --- 4d. 收集数据用于生成最终的CSV文件 ---
        results_data.append({
            "video_name": video_name,
            "instruction": instruction,
            "full_prompt_path": prompt_filepath,
        })

        print(f"Finished processing {video_name}. Cleaning up resources...")
        del video_sg_builder  # 显式删除大的对象
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- 5. 所有任务完成后，生成CSV文件 ---
    print("\n" + "=" * 50)
    print("All videos processed. Generating final CSV file...")
    df = pd.DataFrame(results_data)
    df.to_csv(args.output_csv_path, index=False, encoding='utf-8')
    print(f"Successfully saved results to {args.output_csv_path}")
    print("=" * 50)


if __name__ == '__main__':
    main()