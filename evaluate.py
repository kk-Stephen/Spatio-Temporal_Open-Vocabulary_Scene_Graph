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



def calculate_plan_similarity(predicted_plan: list[str], ground_truth_label: list[str], sentence_model) -> float:
    """
    使用句子嵌入和匈牙利算法计算两个行动计划之间的语义相似度。

    Args:
        predicted_plan (list[str]): 模型生成的行动计划列表。
        ground_truth_label (list[str]): 标注的行动计划列表。
        sentence_model: 用于编码的Sentence Transformer模型。

    Returns:
        一个0到1之间的相似度分数。
    """
    if not predicted_plan or not ground_truth_label:
        return 0.0

    # 1. 将两个计划的每一步都编码成向量
    pred_embeddings = sentence_model.encode(predicted_plan, convert_to_tensor=True)
    label_embeddings = sentence_model.encode(ground_truth_label, convert_to_tensor=True)

    # 2. 计算两个向量集合之间的余弦相似度矩阵
    # similarity_matrix[i][j] 表示 pred_embeddings[i] 和 label_embeddings[j] 的相似度
    similarity_matrix = util.cos_sim(pred_embeddings, label_embeddings).cpu().numpy()

    # 3. 使用匈牙利算法找到最佳匹配
    # 我们希望最大化相似度，而匈牙利算法是最小化成本，所以成本 = 1 - 相似度
    cost_matrix = 1 - similarity_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # 4. 计算最终分数
    # 将所有匹配上的步骤的相似度相加
    matched_similarity_sum = similarity_matrix[row_ind, col_ind].sum()

    # 归一化：用匹配的相似度总和除以较长计划的步骤数，使分数在0-1之间
    # 这样可以公平地处理步骤数不匹配的情况
    normalization_factor = max(len(predicted_plan), len(ground_truth_label))
    final_score = matched_similarity_sum / normalization_factor if normalization_factor > 0 else 0.0

    return final_score


def parse_action_plan_from_llm_output(llm_output: List[str]) -> Optional[str]:
    """
    从LLM的原始输出中解析出 "action_plan" 的内容，并以字符串形式返回。

    该函数能处理包含在Markdown代码块 (```json) 中的JSON。

    Args:
        llm_output (List[str]): LLM模型返回的原始输出，通常是一个包含单个字符串的列表。

    Returns:
        Optional[str]: 成功解析出的 action_plan 列表的字符串表示形式。
                       如果解析失败或找不到 action_plan，则返回 None。
                       例如: '["action 1", "action 2"]'
    """
    # 1. 输入校验：确保输入是列表且不为空
    if not llm_output or not isinstance(llm_output, list) or not llm_output[0]:
        print("Error: Input is empty or not in the expected format (List[str]).")
        return None

    raw_string = llm_output[0]

    # 2. 使用正则表达式提取JSON字符串
    #    这个正则表达式会优先寻找被 ```json ... ``` 包围的部分。
    #    如果找不到，它会回退到寻找任何被 { ... } 包围的部分。
    #    re.DOTALL 标志让 '.' 可以匹配包括换行符在内的任何字符。
    match = re.search(r"```json\s*(\{.*?\})\s*```", raw_string, re.DOTALL)
    if not match:
        # 如果没有找到Markdown代码块，尝试直接在字符串中寻找JSON对象
        match = re.search(r"(\{.*\})", raw_string, re.DOTALL)

    if not match:
        print("Error: Could not find a JSON block in the LLM output.")
        return None

    # 提取括号中匹配到的纯JSON字符串
    json_string = match.group(1)

    # 3. 解析JSON字符串并提取 "action_plan"
    try:
        # 将纯JSON字符串转换为Python字典
        data = json.loads(json_string)

        # 从字典中安全地获取 "action_plan" 的值（一个Python列表）
        action_plan_list = data.get("action_plan")

        if action_plan_list is None:
            print("Error: JSON was parsed, but the 'action_plan' key was not found.")
            return None

        # 4. 将提取出的Python列表转换回JSON格式的字符串
        #    这正是您想要的最终格式，例如 '["action 1", "action 2"]'
        return json.dumps(action_plan_list, ensure_ascii=False)

    except json.JSONDecodeError as e:
        print(f"Error: Failed to parse the extracted JSON string. Error: {e}")
        return llm_output

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
        label_str = row['label']

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

        # b. 解析 Label
        ground_truth_label = label_str.split(',')
        VLA.eval()
        robot_plan = VLA.generate(messages)
        # print("robot_plan", robot_plan)

        # js_plan = parse_action_plan_from_llm_output(robot_plan)
        # if js_plan is not None and js_plan != []:
        #     robot_plan = js_plan
        # print("ground_truth_label",ground_truth_label)
        # c. 计算相似度
        similarity_score = calculate_plan_similarity(robot_plan, ground_truth_label, sentence_model)

        evaluation_results.append({
            "video_name": video_name,
            "instruction_id": row['instruction_id'],
            "instruction": instruction,
            "predicted_plan": robot_plan,
            "ground_truth_label": label_str,
            "similarity_score": similarity_score
        })

        if len(evaluation_results) >= 50:
            save_batch(evaluation_results, args.output_csv_path)

        if torch.cuda.is_available():
            # print("==================cache clean===============")
            torch.cuda.empty_cache()

    # --- 4. 保存评估结果 ---
    if evaluation_results:
        save_batch(evaluation_results, args.output_csv_path)

    results_df = pd.read_csv(args.output_csv_path)

    # 计算并打印平均分
    avg_score = results_df['similarity_score'].mean()

    print("\n" + "=" * 50)
    print(f"Evaluation Complete!")
    print(f"Average Plan Similarity Score: {avg_score:.4f}")
    print("=" * 50)

    results_df.to_csv(args.output_csv_path, index=False, encoding='utf-8')
    print(f"Evaluation results saved to {args.output_csv_path}")


if __name__ == '__main__':
    main()