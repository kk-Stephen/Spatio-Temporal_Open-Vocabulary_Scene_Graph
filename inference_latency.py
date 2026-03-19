# evaluate_planner.py

import pandas as pd
import torch
import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from scipy.optimize import linear_sum_assignment
from util.utils import get_model, get_rgb_depth_paths
import re
from typing import List, Optional



def save_batch(batch, out_csv):
    """将一批结果追加写入 CSV。第一次写入带表头，之后不再写表头。"""
    if not batch:
        return
    df_batch = pd.DataFrame(batch)
    df_batch.to_csv(
        out_csv,
        index=False,
        encoding='utf-8',
        mode='a',
        header=not os.path.exists(out_csv)  # 仅首次写入表头
    )
    batch.clear()

def main():
    parser = argparse.ArgumentParser(description="Evaluate LVLM Planner against ground truth labels.")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to the input CSV file (e.g., planning_dataset.csv).")
    parser.add_argument("--video_dir", type=str, required=True,
                        help="Path to the main directory containing video .mp4 files.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct",
                        help="Name of the Qwen Planner model.")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run models on ('cuda' or 'cpu').")
    parser.add_argument("--output_csv_path", type=str, default="evaluation_results.csv",
                        help="Path to save the final evaluation results CSV file.")

    args = parser.parse_args()

    # --- 1. 初始化和模型加载 ---
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using Device: {device}")

    # 加载用于评估的句子相似度模型
    print("Loading Sentence Transformer model for evaluation...")
    # 'all-MiniLM-L6-v2' 是一个非常快速且效果良好的模型
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    print("Sentence Transformer model loaded.")

    # 加载您的Qwen Planner模型 (此处为占位符)
    print(f"Loading Qwen Planner model: {args.model_name}... (Placeholder)")
    # from util.utils import get_model # 假设您的加载函数在这里
    # qwen_planner_model, _ = get_model(args, device)
    VLA, vla_criterion = get_model(args, device)

    # --- 2. 加载数据集 ---
    df = pd.read_csv(args.csv_path)
    print(f"Loaded {len(df)} tasks from {args.csv_path}")

    # --- 3. 遍历数据集进行评估 ---
    evaluation_results = []
    # for index, row in tqdm(df.head(10).iterrows(), total=min(10, df.shape[0]), desc="Evaluating plans"):
    # for index, row in tqdm(df.iloc[200:].iterrows(), total=df.shape[0] - 200, desc="Evaluating plans"):
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Evaluating plans"):
        video_name = row['video_name']
        instruction = row['instruction']
        prompt_path = row['full_prompt_path']

        video_path = os.path.join(args.video_dir, f"{video_name}.mp4")
        # video_path = os.path.join(args.video_dir, video_name)
        if not os.path.exists(prompt_path) or not os.path.exists(video_path):
            print(f"Warning: Skipping row {index} due to missing prompt or video file.")
            continue
        # rgb_paths, depth_paths = get_rgb_depth_paths(directory=video_path)
        # 读取 prompt 文件
        max_len = 50 *1024
        if os.path.getsize(prompt_path) > max_len:
            print("length exceed max length!!")
            continue

        with open(prompt_path, 'r', encoding='utf-8') as f1, open(os.path.join(os.path.dirname(__file__), 'prompts', 'extra.txt'), 'r', encoding='utf-8') as f2:
            prompt2 = f2.read()
            prompt1 = f1.read()
            prompt = prompt1 + "\n" + prompt2   # 这里用换行符拼接，也可以改成别的


        content_list = []
        # for path in rgb_paths:  # 使用您已有的 rgb_paths 列表
        #     content_list.append({"type": "image", "image": path})

        content_list.append({"type": "video", "video": video_path})

        # 2. 将文本提示追加到列表末尾
        content_list.append({"type": "text", "text": prompt})

        # print("video_path", video_path)
        # print("prompt_path", prompt_path)

        messages = [
            {
                "role": "user",
                "content": content_list,
            }
        ]

        VLA.eval()
        robot_plan = VLA.generate(messages)
        # print("robot_plan", robot_plan)

        # js_plan = parse_action_plan_from_llm_output(robot_plan)
        # if js_plan is not None and js_plan != []:
        #     robot_plan = js_plan
        # print("ground_truth_label",ground_truth_label)
        # c. 计算相似度

        evaluation_results.append({
            "video_name": video_name,
            "instruction": instruction,
            "predicted_plan": robot_plan
        })

        if len(evaluation_results) >= 50:
            save_batch(evaluation_results, args.output_csv_path)

        if torch.cuda.is_available():
            # print("==================cache clean===============")
            torch.cuda.empty_cache()

    # --- 4. 保存评估结果 ---
    if evaluation_results:
        save_batch(evaluation_results, args.output_csv_path)

    print(f"Evaluation results saved to {args.output_csv_path}")


if __name__ == '__main__':
    main()