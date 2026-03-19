import sys
import os
import torch
import numpy as np
import pandas as pd
import argparse
from util.utils import get_model, build_cfg_from_yaml, get_rgb_depth_paths
from scene_graph import create_video_scene_graph
from planner import generate_prompt
from preprocessing import preprocess
import open3d as o3d
from types import SimpleNamespace

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--planner_model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--video_path", type=str, required=True, help="path to video file")
    parser.add_argument("--checkpoint_dir", type=str, help="Path to checkpoint directory")
    parser.add_argument("--frames_dir", type=str, required=True, help="Path of the preprocessed frame directory")
    parser.add_argument("--fps", type=int, required=True, help="Frame rate")
    parser.add_argument("--latency", type=float, required=True, help="Latency")
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--save_prompt_dir", type=str, required=True, default="full_prompt.txt")
    parser.add_argument("--com_ts", type=float, required=True, help="communication time stamp")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    #preprocess(input_path=args.video_path, output_dir=args.frames_dir, fps=args.fps)
    rgb_paths, depth_paths = get_rgb_depth_paths(directory=args.frames_dir)

    # print('rgb_paths', rgb_paths)
    # print('depth_paths', depth_paths)

    LVLM, criterion = get_model(args, device)
    # constuct grounded DINO
    sys.path.insert(0, os.path.abspath("Grounded_SAM_2_main"))
    dino_cfg = build_cfg_from_yaml("configs/Grounded_DINO.yaml")
    Ground_DINO, criterion = get_model(dino_cfg, device)

    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(640, 480, 517.3, 516.5, 318.6, 255.3)
    depth_scale = 5000.0
    depth_trunc = 5.0
    video_SG = create_video_scene_graph(LVLM=LVLM, Ground_DINO=Ground_DINO, frame_dir=args.frames_dir, rgb_paths=rgb_paths, depth_paths=depth_paths,
                                        camera_intrinsics=camera_intrinsics, depth_scale=depth_scale, depth_trunc=depth_trunc, latency=args.latency)
    full_prompt = generate_prompt(video_SG, user_prompt=args.prompt, com_ts=args.com_ts)
    print("full prompt", full_prompt)
    with open(args.save_prompt_dir, "w", encoding="utf-8") as f:
        f.write(full_prompt)

    # vla_args = SimpleNamespace(**vars(args))
    # vla_args.model_name = args.planner_model
    #
    # VLA, vla_criterion = get_model(vla_args, device)
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "video",
    #                 "video": rgb_paths,
    #             },
    #             {
    #                 "type": "text",
    #                 "text": full_prompt
    #             },
    #         ],
    #     }
    # ]
    # VLA.eval()
    # robot_plan = VLA.generate(messages)
    # print("robot_plan", robot_plan)

if __name__ == "__main__":
    main()