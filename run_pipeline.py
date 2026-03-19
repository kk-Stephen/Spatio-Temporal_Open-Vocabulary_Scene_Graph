# main.py

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
    parser.add_argument("--label_dir", type=str, required=True,
                        help="Path to the directory containing corresponding .json label files.")
    parser.add_argument("--output_prompt_dir", type=str, default="full_prompts",
                        help="Directory to save the generated full_prompt .txt files.")
    parser.add_argument("--output_csv_path", type=str, default="planning_dataset.csv",
                        help="Path to save the final output CSV file.")
    parser.add_argument("--fps", type=int, default=2, help="Frame rate of the videos.")
    parser.add_argument("--latency", type=float, default=0.0, help="Assumed system latency in seconds.")
    parser.add_argument("--com_ts", type=float, default=8.0, help="Default communication timestamp for planning.")

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

    # --- 3. 初始化CSV数据收集器 ---
    results_data = []

    # --- 4. 外层循环：遍历所有视频 ---
    video_subfolders = sorted([d for d in os.listdir(args.video_dir) if os.path.isdir(os.path.join(args.video_dir, d))])

    for video_name in video_subfolders:
        print("\n" + "=" * 50)
        print(f"Processing Video: {video_name}")
        print("=" * 50)

        idx = video_name.split("_")[-1]
        json_label_path = os.path.join(args.label_dir, f"label_{idx}.json")
        current_video_dir = os.path.join(args.video_dir, video_name)


        # 检查对应的标签文件是否存在
        if not os.path.exists(json_label_path):
            print(f"Warning: Label file not found for {video_name} at {json_label_path}. Skipping.")
            continue

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
            latency=args.latency
        )
        print("Scene Graph built successfully.")

        # --- 4b. 内层循环：遍历当前视频的所有指令 ---
        print(f"Step 2: Generating prompts for all instructions in {video_name}.json...")
        with open(json_label_path, 'r', encoding='utf-8') as f:
            instructions_data = json.load(f)

        for instruction_id, item in enumerate(instructions_data):
            instruction_text = item.get("instruction")
            action_label = item.get("action")

            if not instruction_text or not action_label:
                continue

            print(f"  -> Processing Instruction #{instruction_id}: '{instruction_text}'")

            # --- 4c. 生成并保存 Full Prompt ---
            full_prompt = generate_prompt(video_sg_builder, user_prompt=instruction_text, com_ts=args.com_ts)

            prompt_filename = f"{video_name}_inst_{instruction_id}.txt"
            prompt_filepath = os.path.join(args.output_prompt_dir, prompt_filename)

            with open(prompt_filepath, "w", encoding="utf-8") as f:
                f.write(full_prompt)


            # --- 4d. 收集数据用于生成最终的CSV文件 ---
            results_data.append({
                "video_name": video_name,
                "instruction_id": instruction_id,
                "instruction": instruction_text,
                "full_prompt_path": prompt_filepath,
                "label": ", ".join(action_label)  # 将action列表用'|'连接成字符串
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