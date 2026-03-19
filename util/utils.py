import importlib
import yaml
from types import SimpleNamespace
import os
import glob

# def get_model(cfg, device):
#     if "Qwen2.5" in cfg.model_name or "Qwen2.5" in cfg.planner_model:
#         module = importlib.import_module("models.qwen25_vl")
#     else:
#         module = importlib.import_module(cfg.MODEL.FILE) #等价于import models.resnet as module
#
#     model, criterion= getattr(module, 'build_model')(cfg, device) #导入biuld_model函数
#
#     return model, criterion

def get_model(cfg, device):
    # 兼容 argparse.Namespace 或 dict
    model_name = getattr(cfg, "model_name", None) or (cfg.get("MODEL", {}).get("NAME") if isinstance(cfg, dict) else None)
    planner_model = getattr(cfg, "planner_model", None)

    # 判断要不要加载 Qwen
    if (model_name and "Qwen2.5" in model_name) or (planner_model and "Qwen2.5" in planner_model):
        module = importlib.import_module("models.qwen25_vl")
    else:
        if isinstance(cfg, dict):
            module_file = cfg["MODEL"]["FILE"]
        else:
            module_file = cfg.MODEL.FILE
        module = importlib.import_module(module_file)

    model, criterion = getattr(module, "build_model")(cfg, device)
    return model, criterion

def build_cfg_from_yaml(yaml_path: str):
    """
    从一个形如：

    MODEL:
      FILE: models.grounded_dino
      NAME: Grounded_DINO
      SAM2_CHECKPOINT: "…"
      …
    的 YAML 文件，创建一个 cfg 对象：
    cfg.MODEL.FILE, cfg.MODEL.NAME, … 可直接访问，
    并且顶层增加 cfg.model_name 方便外部判断。
    """
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    model_dict = data.get("MODEL", {})
    if not model_dict:
        raise ValueError(f"No MODEL section found in {yaml_path}")

    cfg = SimpleNamespace()
    cfg.MODEL = SimpleNamespace(**model_dict)
    cfg.model_name = cfg.MODEL.NAME

    return cfg

from pathlib import Path

def _idx(p: str) -> int:
    # 文件名形如 12_rgb.png / 12_depth.png
    return int(Path(p).stem.split('_')[0])

def get_rgb_depth_paths(directory):
    directory = os.path.abspath(directory)
    rgb_paths   = sorted(glob.glob(os.path.join(directory, "*_rgb.png")),   key=_idx)
    depth_paths = sorted(glob.glob(os.path.join(directory, "*_depth.png")), key=_idx)

    # 可选：确保一一对应（如果有缺帧，自动求交集对齐）
    rgb_map   = {_idx(p): p for p in rgb_paths}
    depth_map = {_idx(p): p for p in depth_paths}
    common = sorted(set(rgb_map) & set(depth_map))
    rgb_paths   = [rgb_map[i]   for i in common]
    depth_paths = [depth_map[i] for i in common]

    return rgb_paths, depth_paths