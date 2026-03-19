import cv2
import torch
import numpy as np
import open3d as o3d
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from dataclasses import dataclass, field
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import os
import glob
from collections import defaultdict
from pycocotools import mask as maskUtils
from pydantic import BaseModel, Field
from typing import Optional, List
import re
from itertools import product
import uuid
import json

def _sanitize_json_like(s: str) -> str:
    # 去 Markdown 围栏
    s = re.sub(r"```(?:json)?\s*", "", s)
    s = s.replace("```", "")
    # 去注释
    s = re.sub(r"//.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    # Python -> JSON 字面量
    s = re.sub(r"\bTrue\b", "true", s)
    s = re.sub(r"\bFalse\b", "false", s)
    s = re.sub(r"\bNone\b", "null", s)
    # 特殊数字
    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\b-?Infinity\b", "null", s)
    # 键用单引号 -> 双引号（保守：只替换键）
    s = re.sub(r'(?P<prefix>[{,\s])\'(?P<key>[^\'"]+?)\'\s*:', r'\g<prefix>"\g<key>":', s)
    # 去尾逗号
    s = re.sub(r",\s*([}\]])", r"\1", s)
    return s.strip()

def _try_load_json(text: str):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        fixed = _sanitize_json_like(text)
        return json.loads(fixed)

def parse_llm_json(text: str, frame_id="unknown", log_dir="logs"):
    """从 LLM 输出中容错提取 JSON（dict/list）。失败会把原文落盘并抛错。"""
    os.makedirs(log_dir, exist_ok=True)
    candidates = []
    # ```json ...``` 或 ``` ... ```
    for pat in (r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"):
        candidates += [m.group(1) for m in re.finditer(pat, text)]
    # 退化：花括号/中括号体
    for pat in (r"(\{[\s\S]*?\})", r"(\[[\s\S]*?\])"):
        candidates += [m.group(1) for m in re.finditer(pat, text)]
    if not candidates:
        candidates = [text]

    last_err = None
    for cand in candidates:
        try:
            return _try_load_json(cand)
        except Exception as e:
            last_err = e
            continue

    raw_path = os.path.join(log_dir, f"bad_json_frame_{frame_id}.txt")
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(text)
    raise ValueError(f"Failed to parse LLM JSON for frame {frame_id}. Saved to {raw_path}. Last error: {last_err}")

def dedupe_preserve(seq):
    seen = set()
    out = []
    for x in seq:
        if x not in seen:
            out.append(x); seen.add(x)
    return out

def make_dino_prompt(objects, max_objs=40):
    """
    GroundingDINO 期望：'apple. orange. cube. duck.'（点+空格分隔，末尾有点）
    - 统一小写、去首尾标点、去重、截断数量
    """
    cleaned = []
    for o in objects:
        o = str(o).strip().lower()
        if not o:
            continue
        # 去掉结尾多余符号
        o = o.strip(" .,:;!?\t\n\r")
        # 防止含句点的类名（极少见），保守把句点换空格
        o = o.replace(".", " ")
        # 合并多空格
        o = re.sub(r"\s+", " ", o)
        if o:
            cleaned.append(o)
    cleaned = dedupe_preserve(cleaned)[:max_objs]
    if not cleaned:
        return ""
    return ". ".join(cleaned) + "."

def load_clip_model(DEVICE):
    """加载CLIP模型和预处理器"""
    try:
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32", use_safetensors=True).to(DEVICE)
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        print("CLIP model loaded successfully.")
        return model, processor
    except Exception as e:
        print(f"Could not load CLIP model: {e}. Using placeholders.")
        return None, None

class Entity(BaseModel):
    track_id: int
    class_name: str = Field(alias="class")

class EventItem(BaseModel):
    id: Optional[str] = None
    subject: Entity
    predicate: str
    object: Optional[Entity] = None
    t_start_frame: int
    t_end_frame: int

@dataclass
class ObjectInstance:
    """存储单帧中一个被检测物体实例的所有信息"""
    frame_id: int
    instance_id_in_frame: int
    class_name: str
    mask: np.ndarray
    bbox: np.ndarray
    view_ts: float = None
    track_id: Optional[int] = None

    # 3D 属性
    centroid_3d: np.ndarray = None #done
    size_3d: np.ndarray = None #done
    point_cloud: o3d.geometry.PointCloud = None #done

    # 特征嵌入
    visual_embedding: np.ndarray = None
    semantic_embedding: np.ndarray = None


@dataclass
class TrackedObject:
    """代表一个跨时间追踪的物体"""
    track_id: int
    class_name: str
    instances: list[ObjectInstance] = field(default_factory=list)
    last_seen_frame: int = -1
    is_active: bool = True

    def get_last_instance(self):
        return self.instances[-1] if self.instances else None

# --- 特征提取 ---

def get_clip_embeddings(instance: ObjectInstance, rgb_image_pil: Image, CLIP_MODEL, CLIP_PROCESSOR, DEVICE="cuda"):
    """
    使用CLIP模型为物体实例提取视觉和文本嵌入。
    """
    if CLIP_MODEL is None or CLIP_PROCESSOR is None:
        # 如果模型加载失败，返回随机向量作为占位符
        instance.visual_embedding = np.random.rand(512)
        instance.semantic_embedding = np.random.rand(512)
        return

    # 视觉嵌入 (Visual Embedding)
    masked_image = rgb_image_pil.copy()
    # 扣bbox
    masked_image_np = np.array(masked_image)
    # PIL图像是RGB, OpenCV是BGR
    visual_mask = np.stack([instance.mask] * 3, axis=-1)
    background = np.full_like(masked_image_np, 255)  # 白色背景
    masked_image_np = np.where(visual_mask, masked_image_np, background)
    masked_image_pil = Image.fromarray(masked_image_np)

    inputs = CLIP_PROCESSOR(images=masked_image_pil, return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        image_features = CLIP_MODEL.get_image_features(**inputs)
    instance.visual_embedding = image_features.to(torch.float32).cpu().numpy().ravel()

    # 语义嵌入 (Semantic Embedding)
    text_inputs = CLIP_PROCESSOR(text=[instance.class_name], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        text_features = CLIP_MODEL.get_text_features(**text_inputs)
    instance.semantic_embedding = text_features.to(torch.float32).cpu().numpy().ravel()


# --- 核心流程类 ---

class VideoSceneGraphBuilder:
    def __init__(self, camera_intrinsics, depth_scale, depth_trunc, latency, clip_model, clip_processor, fps=2.0):
        # 记录一下相机参数写在这
        self.intrinsics = camera_intrinsics #相机内参， 焦距fx,fy；主点坐标cx，cy
        self.depth_scale = depth_scale #缩放因子
        self.depth_trunc = depth_trunc #截尾
        self.fps = fps  # [新增] 用于事件时间戳计算

        self.clip_model = clip_model
        self.clip_processor = clip_processor

        # 视频场景图，使用多边有向图以表示时间流和延迟
        # 有三种边：1.同帧中时空关系边；2.同一实体跨帧时间边；3.事件边
        self.vsg = nx.MultiDiGraph()
        # 延迟
        self.latency_sec = latency

        self.tracked_objects = {}  # key: track_id, value: TrackedObject
        self.next_track_id = 0

        # 匹配权重
        self.w_spatial = 0.6
        self.w_visual = 0.4
        self.MAX_SPATIAL_DIST = 1.5  # 米，超过此距离的匹配代价极高
        self.MATCHING_COST_THRESHOLD = 0.7  # 综合代价阈值，超过则不匹配

    @staticmethod
    def node_key(track_id: int, frame: int) -> str:
        """生成唯一的节点标识符"""
        return f"{track_id}@{frame}"

    def add_temporal_edge(self, track_id: int, frame_prev: int, frame_curr: int):
        """添加表示物体运动轨迹的时间边"""
        u = self.node_key(track_id, frame_prev)
        v = self.node_key(track_id, frame_curr)
        if self.vsg.has_node(u) and self.vsg.has_node(v):
            # 使用 key 来确保即使重复调用也只添加一次
            edge_key = f"temp-{u}-{v}"
            self.vsg.add_edge(u, v, key=edge_key, type="temporal", relation="next_frame")

    def add_spatial_edge(self, frame: int, src_track: int, dst_track: int, relation: str, weight: float = 1.0):
        """添加表示同帧内物体空间关系的边"""
        u = self.node_key(src_track, frame)
        v = self.node_key(dst_track, frame)
        if self.vsg.has_node(u) and self.vsg.has_node(v):
            edge_key = f"spa-{u}-{v}-{relation}"
            self.vsg.add_edge(u, v, key=edge_key, type="spatial", relation=relation, weight=weight)

    # def add_event_edge(self, ev: EventItem):
    #     """添加表示交互事件的边"""
    #     ev_id = ev.id or f"ev_{uuid.uuid4().hex[:8]}"
    #
    #     # 找到事件发生时最接近的节点
    #     subj_key = self._find_closest_node_key(ev.subject.track_id, ev.t_start_frame, ev.t_end_frame)
    #
    #     if ev.object is not None:
    #         obj_key = self._find_closest_node_key(ev.object.track_id, ev.t_start_frame, ev.t_end_frame)
    #     else:  # 如果事件没有宾语 (例如 "人正在走路")
    #         obj_key = subj_key
    #
    #     if subj_key and obj_key:
    #         edge_key = f"evt-{ev_id}"
    #         self.vsg.add_edge(
    #             subj_key, obj_key,
    #             key=edge_key,
    #             type="event",
    #             predicate=ev.predicate,
    #             event_id=ev_id,
    #             t_start_frame=ev.t_start_frame,
    #             t_end_frame=ev.t_end_frame
    #         )
    #         print(f"  Added Event Edge: ({subj_key}) --[{ev.predicate}]--> ({obj_key})")

    def process_frame(self, frame_id, rgb_path, depth_path, results, spati_rel):
        """处理单个视频帧，更新视频场景图"""
        print(f"\n--- Processing Frame {frame_id} ---")

        # 1. 加载图像
        rgb_image = cv2.imread(rgb_path)
        rgb_image_pil = Image.open(rgb_path).convert("RGB")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        # 2. detect & seg results
        seg_bbox_list = results

        # 3. 3D信息融合与特征提取
        current_frame_instances = self._create_instances_for_frame(frame_id, seg_bbox_list, rgb_image, depth_image,
                                                                   rgb_image_pil)

        # 4. 跨帧对象匹配
        matches, new_instances, lost_track_ids = self._match_instances_to_tracks(current_frame_instances)

        # 5. 更新追踪器和场景图
        tracked_instances_this_frame = []
        # a. 更新匹配上的track
        for track_id, instance in matches:
            instance.track_id = track_id
            node_id = self.node_key(track_id, frame_id)
            prev_frame = self.tracked_objects[track_id].last_seen_frame
            instance.view_ts = frame_id/self.fps + self.latency_sec

            self.tracked_objects[track_id].instances.append(instance)
            self.tracked_objects[track_id].last_seen_frame = frame_id
            self.tracked_objects[track_id].is_active = True

            # 添加实例节点和时间边
            self.vsg.add_node(node_id, **instance.__dict__)
            self.add_temporal_edge(track_id, prev_frame, frame_id)
            tracked_instances_this_frame.append(instance)
            print(f"  Matched: Track {track_id} -> Instance {instance.instance_id_in_frame} ({instance.class_name})")

        # b. 为未匹配的新实例创建新track
        for instance in new_instances:
            new_track_id = self.next_track_id
            self.next_track_id += 1
            instance.view_ts = frame_id / self.fps + self.latency_sec

            new_tracked_obj = TrackedObject(track_id=new_track_id, class_name=instance.class_name, instances=[instance],
                                            last_seen_frame=frame_id)
            self.tracked_objects[new_track_id] = new_tracked_obj
            instance.track_id = new_track_id

            node_id = self.node_key(new_track_id, frame_id)
            self.vsg.add_node(node_id, **instance.__dict__)
            tracked_instances_this_frame.append(instance)
            print(
                f"  New Track: Track {new_track_id} <- Instance {instance.instance_id_in_frame} ({instance.class_name})")

        # c. 标记丢失的track
        for track_id in lost_track_ids:
            self.tracked_objects[track_id].is_active = False
            print(f"  Lost Track: Track {track_id} ({self.tracked_objects[track_id].class_name})")

        self._infer_and_add_spatial_relations(frame_id, tracked_instances_this_frame, spati_rel)

    def _create_instances_for_frame(self, frame_id, detections, rgb_image_bgr, depth_image, rgb_image_pil):
        """为单帧的检测结果创建ObjectInstance对象列表"""
        instances = []

        H, W = depth_image.shape[:2]
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image_bgr, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d,
            depth_scale=self.depth_scale,
            depth_trunc=self.depth_trunc,
            convert_rgb_to_intensity=False)
        full_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, self.intrinsics)

        # ---- 2) 构建像素→点云索引映射（关键）----
        # Open3D 会丢弃无效深度(0或>trunc)，且按行主序遍历有效像素生成点的顺序
        if np.issubdtype(depth_image.dtype, np.floating):
            valid_mask = (depth_image > 0) & (depth_image <= self.depth_trunc + 1e-12)
        else:
            # 深度是整数（例如 TUM: uint16 毫米；RealSense: uint16 原始单位）
            depth_units_trunc = int(round(self.depth_trunc * self.depth_scale))
            valid_mask = (depth_image > 0) & (depth_image <= depth_units_trunc)

            # valid 像素的线性下标（行主序）
            valid_lin_idx = np.flatnonzero(valid_mask.ravel())
            # full_pcd 中点的数量应该与有效像素一致
            n_valid = valid_lin_idx.size
            if len(full_pcd.points) != n_valid:
                print(f"Warning: Discrepancy found between calculated valid pixels ({n_valid}) and points in PointCloud ({len(full_pcd.points)}). Applying conservative adjustment.")
                # 若不一致，说明你的 valid_mask 判定和 Open3D 内部略有差异；通常是阈值边界或NaN。
                # 这里做一次保守收紧（仅作为兜底）
                n_pcd = len(full_pcd.points)
                if n_pcd < n_valid:
                    valid_lin_idx = valid_lin_idx[:n_pcd]
                    n_valid = n_pcd
                else:
                    # 极少见：Open3D 可能还有其它过滤；通常不会大于 n_valid，此处仅防御
                    pass

            # 像素索引 -> 点云索引（-1 表示该像素没有对应点）
            pixel2pcd = np.full(H * W, -1, dtype=np.int32)
            pixel2pcd[valid_lin_idx] = np.arange(n_valid, dtype=np.int32)

        # ---- 3) 遍历实例：用 mask 从 full_pcd 里“切”出对应点 ----
        for i, det in enumerate(detections):
            cls = det["class_name"]
            seg = det["segmentation"]
            bbox = det["bbox"]

            # RLE -> bool mask
            mask = self.decode_coco_rle(seg)
            if mask is None:
                continue
            mask = np.asarray(mask, dtype=bool)

            # 若 mask 尺寸与帧不一致（很少），做 NN 缩放
            if mask.shape != (H, W):
                mask = cv2.resize(mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)

            # 由像素选点：先把 mask 的线性索引找出来，再查表映射到点云索引
            obj_pixels = np.flatnonzero(mask.ravel())
            if obj_pixels.size == 0:
                continue
            obj_pcd_idx = pixel2pcd[obj_pixels]
            obj_pcd_idx = obj_pcd_idx[obj_pcd_idx != -1]  # 去掉无效深度位置
            if obj_pcd_idx.size == 0:
                continue

            # 从整帧点云中选择该实例的点
            obj_pcd = full_pcd.select_by_index(obj_pcd_idx.tolist())
            if not obj_pcd.has_points():
                continue

            # 组装实例
            instance = ObjectInstance(
                frame_id=frame_id,
                instance_id_in_frame=i,
                class_name=cls,
                mask=mask,
                bbox=np.array(bbox) if bbox is not None else None,
            )
            instance.point_cloud = obj_pcd
            instance.centroid_3d = obj_pcd.get_center()

            # OBB
            obb = obj_pcd.get_oriented_bounding_box()
            instance.size_3d = tuple(map(float, obb.extent))

            # 特征提取
            get_clip_embeddings(instance, rgb_image_pil, self.clip_model, self.clip_processor)
            instances.append(instance)

        return instances

    @staticmethod
    def decode_coco_rle(rle_obj):
        """
        rle_obj: {"size":[h,w], "counts": <rle_string>}
        返回: bool mask (h, w)
        """
        # pycocotools 接受 dict 直接 decode
        m = maskUtils.decode(rle_obj)  # uint8 {0,1}, shape (h,w,1) 或 (h,w)
        if m.ndim == 3:
            m = m[..., 0]
        # check the output
        return m.astype(bool)

    @staticmethod
    def _is_valid_bbox(b):
        if b is None:
            return False
        b = np.asarray(b)
        # 允许 (4,), (1,4), (4,1) 等形状
        if b.size != 4:
            return False
        return np.all(np.isfinite(b))

    @staticmethod
    def _calculate_spatial_cost(subj_inst, obj_inst):
        # 提取 bbox
        subj_bbox = np.asarray(subj_inst.bbox, dtype=float)  # [x1,y1,x2,y2]
        obj_bbox = np.asarray(obj_inst.bbox, dtype=float)

        # 计算中心点 ( (x1+x2)/2, (y1+y2)/2 )
        subj_center = ((subj_bbox[0] + subj_bbox[2]) / 2.0,
                       (subj_bbox[1] + subj_bbox[3]) / 2.0)
        obj_center = ((obj_bbox[0] + obj_bbox[2]) / 2.0,
                      (obj_bbox[1] + obj_bbox[3]) / 2.0)

        # 欧几里得距离
        dist = np.linalg.norm(np.array(subj_center) - np.array(obj_center))
        return dist
    def _match_instances_to_tracks(self, current_instances):
        """使用匈牙利算法进行跨帧匹配"""
        active_tracks = [t for t in self.tracked_objects.values() if t.is_active]

        if not active_tracks or not current_instances:
            return [], current_instances, [t.track_id for t in active_tracks]

        # 构建代价矩阵
        num_tracks = len(active_tracks)
        num_instances = len(current_instances)
        cost_matrix = np.full((num_tracks, num_instances), np.inf)

        for i, track in enumerate(active_tracks):
            for j, instance in enumerate(current_instances):
                # 空间距离代价 (0 to 1)
                dist = np.linalg.norm(track.get_last_instance().centroid_3d - instance.centroid_3d)
                spatial_cost = min(dist / self.MAX_SPATIAL_DIST, 1.0)

                # 视觉相似度代价 (0 to 1)
                vec1 = track.get_last_instance().visual_embedding
                vec2 = instance.visual_embedding
                cos_sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                visual_cost = 1 - cos_sim

                # 综合代价
                total_cost = self.w_spatial * spatial_cost + self.w_visual * visual_cost

                # 语义约束：如果类别名不同，增加一个大的惩罚项，但不是无穷大，以允许类别名修正
                if track.class_name != instance.class_name:
                    total_cost += 0.5

                cost_matrix[i, j] = total_cost

        # 匈牙利算法求解最优匹配
        track_indices, instance_indices = linear_sum_assignment(cost_matrix)

        # 解析匹配结果
        matches = []
        matched_instance_ids = set()
        for track_idx, inst_idx in zip(track_indices, instance_indices):
            cost = cost_matrix[track_idx, inst_idx]
            if cost < self.MATCHING_COST_THRESHOLD:
                track_id = active_tracks[track_idx].track_id
                instance = current_instances[inst_idx]
                matches.append((track_id, instance))
                matched_instance_ids.add(inst_idx)

        unmatched_instances = [inst for i, inst in enumerate(current_instances) if i not in matched_instance_ids]

        matched_track_ids = {active_tracks[i].track_id for i, j in zip(track_indices, instance_indices) if
                             cost_matrix[i, j] < self.MATCHING_COST_THRESHOLD}
        lost_track_ids = [t.track_id for t in active_tracks if t.track_id not in matched_track_ids]

        return matches, unmatched_instances, lost_track_ids

    def _calculate_bbox_iou(self, boxA: list[int], boxB: list[int]) -> float:
        """计算两个边界框的交集（IoU）。"""
        # Determine the coordinates of the intersection rectangle
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        # Compute the area of intersection
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        if interArea == 0:
            return 0.0

        # Compute the area of both bounding boxes
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

        # Compute the IoU
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou

    def pair_cost(self, s, o, zone):
        sb, ob = s.bbox, o.bbox
        union = [min(sb[0], ob[0]), min(sb[1], ob[1]), max(sb[2], ob[2]), max(sb[3], ob[3])]
        iou_u = self._calculate_bbox_iou(union, zone) if zone is not None else 0.5

        # 面积一致性（避免“用大盒子搏高IoU”）
        area_union = (union[2] - union[0]) * (union[3] - union[1])
        area_zone = (zone[2] - zone[0]) * (zone[3] - zone[1]) if zone is not None else area_union
        area_ratio = area_union / max(1.0, area_zone)
        area_pen = abs(np.log(area_ratio))  # 接近1最好，偏大/偏小都罚

        # 中心偏移（辅助项，归一化到 zone 对角线）
        def center(b): return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
        cz = center(zone) if zone is not None else center(union)
        cu = center(union)
        diag_z = max(1e-6, np.hypot(zone[2] - zone[0], zone[3] - zone[1])) if zone is not None else \
            max(1e-6, np.hypot(union[2] - union[0], union[3] - union[1]))
        center_shift = np.linalg.norm(np.array(cu) - np.array(cz)) / diag_z

        # 质量自适应权重：zone 越可信，union 权重越大
        # 这里用 iou_u 粗当“zone 可信度”，也可以用 LVLM 置信度
        #alpha = 0.7 + 0.3 * np.clip(iou_u, 0, 1)  # ∈[0.7,1.0]
        #w_u, w_area, w_c, w_d = alpha, 0.4, (1 - alpha), 0.6
        w_u, w_a, w_c = 1.0, 0.4, 0.6

        cost = w_u * (1 - iou_u) + w_a * area_pen + w_c * center_shift
        return cost

    def _infer_and_add_spatial_relations(self, frame_id: int, instances: list, lvlm_output: list):
        spat_rel = lvlm_output
        # print("spat_rel",spat_rel)
        if len(instances) < 2 or not spat_rel:
            return

        # 2. 将当前帧的实例按类别名进行分组，方便查找
        instances_by_class = defaultdict(list)
        for inst in instances:
            # print("inst.class_name", inst.class_name)
            instances_by_class[inst.class_name].append(inst)

        IOU_THRESHOLD = 0.01  # 实例必须与zone至少有10%的重叠

        # 3. 遍历LVLM提出的每个空间关系
        for relation_item in spat_rel:
            #提取关系和bbox
            relation_str = relation_item['relation']
            zone_bbox = relation_item['bbox']
            if zone_bbox is None:
                continue


            # 3.1 解析关系字符串，例如 "<cup> on <table>"
            match = re.match(r"^\s*<([^>]+)>\s+(.+?)\s+<([^>]+)>\s*$", relation_str)
            if not match:
                continue
            # print("match",match)
            print("zone_bbox", zone_bbox)
            subj_class, predicate, obj_class = match.groups()

            # 3.2获得所有候选物体
            all_subj_candidates = instances_by_class.get(subj_class, [])
            all_obj_candidates = instances_by_class.get(obj_class, [])
            # for inst in all_subj_candidates:
            #     print("inst.all_subj_candidates", inst.class_name)
            # for inst in all_obj_candidates:
            #     print("inst.all_obj_candidates", inst.class_name)
            if not all_subj_candidates or not all_obj_candidates:
                continue

            #3.4 消除歧义
            best_pair = None
                # Case 1: Perfect, unambiguous 1-to-1 match
            # print(f"len(filtered_subj) {len(filtered_subj)}; len(filtered_obj) {len(filtered_obj)}")
            if len(all_subj_candidates) == 1 and len(all_obj_candidates) == 1:
                subj_inst = all_subj_candidates[0]
                obj_inst = all_obj_candidates[0]
                # Ensure the single candidates are not the same object
                if subj_inst.track_id != obj_inst.track_id:
                    best_pair = (subj_inst, obj_inst)
                    print(
                        f"  LVLM-Guided Relation: Found 1-to-1 match for '{relation_str}' -> ({subj_inst.track_id}, {obj_inst.track_id})")

            filtered_subj = []
            for inst in all_subj_candidates:
                inst_bbox = inst.bbox
                # print(
                #     f"name: {inst.class_name};bbox: {self._is_valid_bbox(inst_bbox)}; iou: {self._calculate_bbox_iou(inst_bbox, zone_bbox) > IOU_THRESHOLD}")
                if self._is_valid_bbox(inst_bbox) and self._calculate_bbox_iou(inst_bbox,
                                                                               zone_bbox) > IOU_THRESHOLD:
                    filtered_subj.append(inst)

            filtered_obj = []
            for inst in all_obj_candidates:
                # Handle cases where subj and obj are the same class (e.g., <desk> under <desk>)
                # Ensure we don't add the same instance to both lists if they could be the same object
                if inst in filtered_subj and subj_class == obj_class: continue
                inst_bbox = inst.bbox
                # print(
                #     f"name: {inst.class_name};bbox: {self._is_valid_bbox(inst_bbox)}; iou: {self._calculate_bbox_iou(inst_bbox, zone_bbox) > IOU_THRESHOLD}")
                if self._is_valid_bbox(inst_bbox) and self._calculate_bbox_iou(inst_bbox,
                                                                               zone_bbox) > IOU_THRESHOLD:
                    filtered_obj.append(inst)

                # Case 2: 1-to-N ambiguity (one subject, multiple objects). Find the best object.
            if len(filtered_subj) == 1 and len(filtered_obj) > 1:
                subj_inst = filtered_subj[0]
                min_cost = float('inf')
                best_obj_candidate = None
                for obj_inst in filtered_obj:
                    # Skip if it's the same object
                    if subj_inst.track_id == obj_inst.track_id: continue

                    cost = self.pair_cost(subj_inst, obj_inst, zone_bbox)
                    if cost < min_cost:
                        min_cost = cost
                        best_obj_candidate = obj_inst

                if best_obj_candidate:
                    best_pair = (subj_inst, best_obj_candidate)
                    print(
                        f"  LVLM-Guided Relation: Resolved 1-to-N ambiguity for '{relation_str}'. Best match: ({subj_inst.track_id}, {best_obj_candidate.track_id}) with cost {min_cost:.2f}")

                # Case 3: N-to-1 ambiguity (multiple subjects, one object). Find the best subject.
            elif len(filtered_subj) > 1 and len(filtered_obj) == 1:
                obj_inst = filtered_obj[0]
                min_cost = float('inf')
                best_subj_candidate = None
                for subj_inst in filtered_subj:
                    # Skip if it's the same object
                    if subj_inst.track_id == obj_inst.track_id: continue

                    cost = self.pair_cost(subj_inst, obj_inst, zone_bbox)
                    if cost < min_cost:
                        min_cost = cost
                        best_subj_candidate = subj_inst

                if best_subj_candidate:
                    best_pair = (best_subj_candidate, obj_inst)
                    print(
                        f"  LVLM-Guided Relation: Resolved N-to-1 ambiguity for '{relation_str}'. Best match: ({best_subj_candidate.track_id}, {obj_inst.track_id}) with cost {min_cost:.2f}")

                # Case 4: N-to-N ambiguity (multiple subjects, multiple objects). Ignore as requested.
            elif len(filtered_subj) > 1 and len(filtered_obj) > 1:
                print(
                    f"  LVLM-Guided Relation: Skipped '{relation_str}' due to N-to-N ambiguity ({len(filtered_subj)} subjects, {len(filtered_obj)} objects).")
                # best_pair remains None, so we implicitly continue to the next relation.

                # 3.5 If a best pair was found in any of the valid cases, add the edge to the graph
            if best_pair:
                best_subj, best_obj = best_pair
                # The print statements are now inside each specific case for better logging.
                # print("create a spatial edge!!")
                self.add_spatial_edge(frame_id, best_subj.track_id, best_obj.track_id, predicate)

    def visualize_graph(self, output_path="video_scene_graph_detailed.png"):
        """[修改] 增强版可视化，区分不同类型的边"""
        plt.figure(figsize=(24, 24))
        pos = nx.spring_layout(self.vsg, k=0.8, iterations=50, seed=42)

        # 绘制节点
        node_colors = [int(n.split('@')[0]) for n in self.vsg.nodes()]
        nx.draw_networkx_nodes(self.vsg, pos, node_color=node_colors, cmap=plt.cm.viridis, node_size=1200)
        nx.draw_networkx_labels(self.vsg, pos, font_size=9, font_color='white')

        # 分类并绘制边
        edges = self.vsg.edges(keys=True, data=True)
        temporal_edges = [(u, v) for u, v, k, d in edges if d['type'] == 'temporal']
        spatial_edges = [(u, v) for u, v, k, d in edges if d['type'] == 'spatial']
        # event_edges = [(u, v) for u, v, k, d in edges if d['type'] == 'event']

        nx.draw_networkx_edges(self.vsg, pos, edgelist=temporal_edges, edge_color='gray', style='dashed', alpha=0.7,
                               arrows=True)
        nx.draw_networkx_edges(self.vsg, pos, edgelist=spatial_edges, edge_color='blue', style='solid', alpha=0.6,
                               connectionstyle='arc3,rad=0.1')
        # nx.draw_networkx_edges(self.vsg, pos, edgelist=event_edges, edge_color='red', style='solid', width=2.0,
        #                       arrowsize=20)

        # 绘制边的标签
        spatial_labels = {(u, v): d['relation'] for u, v, k, d in edges if d['type'] == 'spatial'}
        # event_labels = {(u, v): d['predicate'] for u, v, k, d in edges if d['type'] == 'event'}

        nx.draw_networkx_edge_labels(self.vsg, pos, edge_labels=spatial_labels, font_color='blue', font_size=8)
        # nx.draw_networkx_edge_labels(self.vsg, pos, edge_labels=event_labels, font_color='red', font_size=10)

        plt.title("Detailed Video Scene Graph (Temporal, Spatial, Event)")
        plt.savefig(output_path)
        plt.close()
        print(f"\nGraph visualization saved to {output_path}")


# --- 主程序 ---
#输入图片序列,每帧的seg_results,每帧的lvlm输出以及相机参数camera_intrinsics, depth_scale
def create_video_scene_graph(LVLM, Ground_DINO, frame_dir, rgb_paths, depth_paths, camera_intrinsics, depth_scale, depth_trunc, latency):
    if not os.path.exists(frame_dir):
        print(f"Error: Data directory '{frame_dir}' not found. Please create it and add your frame data.")
    else:
        clip_model, clip_processor = load_clip_model("cuda")
        builder = VideoSceneGraphBuilder(camera_intrinsics=camera_intrinsics, depth_scale=depth_scale, depth_trunc=depth_trunc,
                                         latency=latency, clip_model=clip_model, clip_processor=clip_processor)
        rgb_files = rgb_paths
        # rgb_files = ["datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031452.959701.png",
        #              "datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031454.791641.png",
        #              "datasets/tum/rgbd_dataset_freiburg1_desk/rgb/1305031452.959701.png"]
        #for rgb_path in rgb_files:
        # print("rgb_files",rgb_files)
        for idx, rgb_path in enumerate(rgb_files):
            # frame_id_str = os.path.basename(rgb_path).split('_')[1]
            frame_id_str = idx
            depth_path = depth_paths[idx]
            # print(f"Processing frame {depth_path}")
           #  print(f"Processing RGB: {rgb_path}")
            # depth_path = "datasets/tum/rgbd_dataset_freiburg1_desk/depth/1305031453.374112.png"

            #通过LVLM物体和关系
            with open(os.path.join(os.path.dirname(__file__), 'prompts', 'frame_detect_prompt.txt'), 'r', encoding='utf-8') as f:
                prompt = f.read()
            messages = [
                {
                    "role": "user",
                    "content": [

                        {"type": "image",
                         "image": rgb_path},
                       {
                            "type": "text",
                            "text": prompt
                        },
                    ],
                }
            ]

            LVLM.eval()
            lvlm_output = LVLM.generate(messages)

            # print("lvlm_output", lvlm_output)
            s = "".join(lvlm_output) if isinstance(lvlm_output, (list, tuple)) else str(lvlm_output)
            m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.S) or re.search(r"(\{.*\})", s, re.S)

            # --- Start of Fix ---
            data = {}  # Initialize with an empty dict
            if m:
                try:
                    data = json.loads(m.group(1))
                except json.JSONDecodeError as e:
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    print(f"WARNING: Frame {frame_id_str} - Failed to parse LVLM JSON output. It might be incomplete.")
                    print(f"Error: {e}")
                    print(f"Incomplete String: {m.group(1)[-100:]}")  # Print the end of the broken string
                    print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    # Continue with an empty 'data' dict, so the script doesn't crash
            else:
                print(f"WARNING: Frame {frame_id_str} - Could not find any JSON block in LVLM output.")
            # --- End of Fix ---

            # objects 列表
            objects = [str(o).strip() for o in data.get("objects", []) if str(o).strip()]

            # This check is now important. If parsing failed, objects will be empty.
            if not objects:
                print("WARNING: No objects found from LVLM output for this frame. Skipping Ground DINO and processing.")
                continue  # Skip to the next frame

            dot_prompt = ".".join(objects) # "computer monitor.keyboard.mouse.desk.book.chessboard"

            # spatial_relations：每项包含 relation 和 bbox
            spatial_relations = []
            for item in data.get("spatial_relation", []):
                if isinstance(item, dict):
                    spatial_relations.append({
                        "relation": item.get("relation"),
                        "bbox": item.get("bbox")
                    })
                else:  # 兼容纯字符串关系
                    spatial_relations.append({"relation": str(item), "bbox": None})

            results = Ground_DINO(dot_prompt, rgb_path)
            seg_results = results["annotations"]
            # print("seg_results",seg_results)
            # print("lvlm_output", lvlm_output)
            if os.path.exists(depth_path):
                builder.process_frame(frame_id=int(frame_id_str), rgb_path=rgb_path, depth_path=depth_path,
                                      results=seg_results, spati_rel=spatial_relations)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        builder.visualize_graph()
        tracked_objects = list(builder.tracked_objects.values())
        for track in tracked_objects:
            # 使用每个被追踪物体最新的实例作为其代表
            last_instance = track.get_last_instance()
            print("tack_id_classname", last_instance.class_name)

        return builder



