import numpy as np
import networkx as nx
import json
from typing import List, Dict, Any
from scene_graph import VideoSceneGraphBuilder, ObjectInstance, TrackedObject
import torch
from collections import defaultdict
# Assume VideoSceneGraphBuilder and its dependencies are defined in the same scope or imported
# from vsg_builder import VideoSceneGraphBuilder, ObjectInstance

def cosine_similarity(vec_a, vec_b):
    """Calculates cosine similarity between two vectors."""
    return np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))


class TaskPlanner:
    """Uses a completed Video Scene Graph to help an LVLM plan robot actions."""

    def __init__(self, vsg_builder: VideoSceneGraphBuilder):
        """
        Initializes the planner with a builder that contains the scene graph and models.

        Args:
            vsg_builder: An instance of VideoSceneGraphBuilder that has already processed video frames.
        """
        self.builder = vsg_builder
        self.vsg = vsg_builder.vsg
        self.clip_model = vsg_builder.clip_model
        self.clip_processor = vsg_builder.clip_processor
        self.fps = vsg_builder.fps
        self.latency = vsg_builder.latency_sec
        self.device = next(self.clip_model.parameters()).device  # Get device from model
        print("TaskPlanner initialized.")

    def _get_text_embedding(self, text: str) -> np.ndarray:
        """Encodes a text string using the loaded CLIP model."""
        inputs = self.clip_processor(text=[text], return_tensors="pt", padding=True).to(self.device)
        with torch.no_grad():
            text_features = self.clip_model.get_text_features(**inputs)
        return text_features.to(torch.float32).cpu().numpy().ravel()

    def find_relevant_entities(self, user_command: str, top_k: int = 5) -> List[str]:
        """
        Finds the most relevant objects in the scene graph for a given user command. not just on the last frame.

        Args:
            user_command: The natural language command from the user.
            latest_frame_id: The most recent frame to consider.
            top_k: The number of most relevant entities to return.

        Returns:
            A list of node keys (e.g., ['5@30', '2@30']) for the top_k most relevant objects.
        """
        command_embedding = self._get_text_embedding(user_command)

        tracked_objects = list(self.builder.tracked_objects.values())
        if not tracked_objects:
            print("警告: 场景图中没有任何被追踪的物体。")
            return []

        scores = []
        track_ids = []

        for track in tracked_objects:
            # 使用每个被追踪物体最新的实例作为其代表
            last_instance = track.get_last_instance()
            if not last_instance:
                continue

            semantic_emb = last_instance.semantic_embedding
            visual_emb = last_instance.visual_embedding
            if semantic_emb is None or visual_emb is None:
                continue

            # 结合语义和视觉特征
            combined_emb = 0.5 * semantic_emb + 0.5 * visual_emb
            score = cosine_similarity(command_embedding, combined_emb)
            scores.append(score)
            track_ids.append(track.track_id)

        if not scores:
            return []

        # Get the indices of the top_k highest scores
        top_k_indices = np.argsort(scores)[-top_k:][::-1]
        top_track_ids = [track_ids[i] for i in top_k_indices]
        # print(f"   找到 Top-{top_k} 相关物体的追踪ID: {top_track_ids}")
        return top_track_ids

    def extract_task_subgraph(self, root_track_ids: List[int]) -> nx.MultiDiGraph:
        """
        Extracts a subgraph around the root nodes, including their neighbors.

        Args:
            root_track_ids: 一个起始物体追踪ID的列表。

        Returns:
            A NetworkX MultiDiGraph representing the task-relevant context.
        """

        if not root_track_ids:
            return nx.MultiDiGraph()

        all_relevant_track_ids = set(root_track_ids)

        # 然后查找根物体在所有帧中的空间邻居，并将它们的track_id也加进来
        related_track_ids = set()
        for track_id in root_track_ids:
            if track_id not in self.builder.tracked_objects: continue
            track = self.builder.tracked_objects[track_id]
            # 遍历该物体的每一个历史实例
            for instance in track.instances:
                node_key = self.builder.node_key(track_id, instance.frame_id)
                if node_key not in self.vsg: continue
                # 查找从该实例出发的所有空间关系边
                for _, neighbor_node_key, edge_data in self.vsg.out_edges(node_key, data=True):
                    if edge_data.get('type') == 'spatial':
                        neighbor_track_id = self.vsg.nodes[neighbor_node_key].get('track_id')
                        if neighbor_track_id is not None:
                            related_track_ids.add(neighbor_track_id)

        all_relevant_track_ids.update(related_track_ids)
        # print(f"   所有相关物体的追踪ID: {all_relevant_track_ids}")

        # 步骤2: 根据所有相关的 track_id 构建子图
        task_subgraph = nx.MultiDiGraph()
        subgraph_node_keys = set()

        # 将这些物体的所有历史实例节点都加入子图
        for track_id in all_relevant_track_ids:
            if track_id not in self.builder.tracked_objects: continue
            track = self.builder.tracked_objects[track_id]
            for instance in track.instances:
                node_key = self.builder.node_key(track_id, instance.frame_id)
                if self.vsg.has_node(node_key):
                    task_subgraph.add_node(node_key, **self.vsg.nodes[node_key])
                    subgraph_node_keys.add(node_key)

        # 步骤3: 添加连接这些节点的所有边 (时间边和空间边)
        for u, v, key, data in self.vsg.edges(keys=True, data=True):
            if u in subgraph_node_keys and v in subgraph_node_keys:
                task_subgraph.add_edge(u, v, key=key, **data)

        # print(f"   子图提取完成，包含 {task_subgraph.number_of_nodes()} 个节点和 {task_subgraph.number_of_edges()} 条边。")
        return task_subgraph

    def graph2json(self, file_path: str):
        """
        [新增] 将完整的视频场景图 (self.vsg) 序列化并保存为 JSON 文件。

        该函数会处理图中的所有节点和边，并将它们组织成以帧为单位的结构，
        最后将结果写入到指定的文件路径。

        Args:
            file_path (str): 保存JSON文件的路径 (例如: "full_scene_graph.json")。
        """
        print(f"\nSerializing full graph to {file_path}...")

        full_graph = self.vsg

        frame_states = defaultdict(lambda: {"nodes": [], "edges": []})
        temporal_edges = []
        other_edges = []

        for node_key, data in full_graph.nodes(data=True):
            frame_id = data.get("frame_id")
            if frame_id is None:
                continue
            frame_key = f"frame_{frame_id}"
            node_info = {
                "node_id": node_key,
                "track_id": data.get("track_id"),
                "class_name": data.get("class_name"),
                "capture_time": float(round(data.get("frame_id") / self.fps if self.fps > 0 else 0.0, 3)),
                "user_observe_time": float(round(data.get("view_ts", 0.0), 3)) if data.get(
                    "view_ts") is not None else None,
                "centroid_3d": [float(round(c, 3)) for c in data.get("centroid_3d", [])],
                "size_3d": [float(round(s, 3)) for s in data.get("size_3d", [])],
                "bbox_2d": [int(b) for b in data.get("bbox", [])] if data.get("bbox") is not None else None,
            }
            frame_states[frame_key]["nodes"].append(node_info)

        for u, v, data in full_graph.edges(data=True):
            edge_type = data.get("type")
            edge_info = {"source": u, "target": v, "type": edge_type, "relation": data.get("relation")}
            if edge_type == "spatial":
                frame_id = full_graph.nodes[u].get("frame_id")
                if frame_id is not None:
                    frame_key = f"frame_{frame_id}"
                    frame_states[frame_key]["edges"].append(edge_info)
            elif edge_type == "temporal":
                temporal_edges.append(edge_info)
            else:
                other_edges.append(edge_info)

        sorted_frame_states = dict(sorted(frame_states.items(), key=lambda item: int(item[0].split('_')[1])))

        full_graph_dict = {
            "metadata": {
                "total_frames": len(sorted_frame_states),
                "total_tracks": self.builder.next_track_id,
                "fps": self.fps,
                "latency": self.latency,
            },
            "frame_states": sorted_frame_states,
            "temporal_edges": temporal_edges,
            "other_edges": other_edges,
        }

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(full_graph_dict, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved full scene graph to {file_path}")
        except Exception as e:
            print(f"Error saving graph to JSON: {e}")

    def serialize_subgraph_to_json(self, subgraph: nx.MultiDiGraph) -> Dict[str, Any]:
        """
        Args:
            subgraph: 从VSG中提取的任务相关子图。

        Returns:
            一个字典，其中 "frame_states" 按帧组织节点和空间边，
            "temporal_edges" 包含连接不同帧的时间边。

        Example Output:
            {
              "frame_states": {
                "frame_30": {
                  "nodes": [ ... ],
                  "edges": [ ... spatial edges in frame 30 ... ]
                },
                "frame_31": {
                  "nodes": [ ... ],
                  "edges": [ ... spatial edges in frame 31 ... ]
                }
              },
              "temporal_edges": [ ... temporal edges connecting frames ... ]
            }
        """

        # 初始化主数据结构
        frame_states = {}
        temporal_edges = []

        # 步骤 1: 遍历所有节点，并按帧进行分组
        for node_key, data in subgraph.nodes(data=True):
            frame_id = data.get("frame_id")
            if frame_id is None:
                continue

            frame_key = f"frame_{frame_id}"

            # 如果字典中还没有这一帧，则初始化
            if frame_key not in frame_states:
                frame_states[frame_key] = {"nodes": [], "edges": []}

            # 构建节点信息
            node_info = {
                "node_id": node_key,
                "track_id": data.get("track_id"),
                "class_name": data.get("class_name"),
                # 将numpy数组转换为列表以便JSON序列化
                "capture_time": float(round(data.get("frame_id") / self.fps if self.fps > 0 else 0.0, 3)),
                "user_observe_time": float(round(data.get("view_ts", 0.0), 3)) if data.get("view_ts") is not None else None,
                # 3D 坐标和尺寸保留 3 位小数，并强制转为 float
                "centroid_3d": [float(round(c, 3)) for c in data.get("centroid_3d", [])],
                "size_3d": [float(round(s, 3)) for s in data.get("size_3d", [])],
            }
            frame_states[frame_key]["nodes"].append(node_info)

        # 步骤 2: 遍历所有边，并将它们分类到对应的帧或时间边列表中
        for u, v, data in subgraph.edges(data=True):
            edge_type = data.get("type")
            edge_info = {
                "source": u,
                "target": v,
                "type": edge_type,
                "relation": data.get("relation"),
            }

            if edge_type == "spatial":
                # 空间边属于单个帧，将其放入对应帧的 "edges" 列表
                # 因为空间边的两个节点在同一帧，所以从任一节点获取frame_id即可
                frame_id = subgraph.nodes[u].get("frame_id")
                if frame_id is not None:
                    frame_key = f"frame_{frame_id}"
                    # 确保该帧的条目已存在
                    if frame_key in frame_states:
                        frame_states[frame_key]["edges"].append(edge_info)

            elif edge_type == "temporal":
                # 时间边连接不同帧，将其放入顶层的 "temporal_edges" 列表
                temporal_edges.append(edge_info)

            # event 边也可以在这里添加分类逻辑
            # elif edge_type == "event": ...

        # 步骤 3: 组装最终的 world_state
        world_state = {
            "frame_states": frame_states,
            "temporal_edges": temporal_edges
        }

        return world_state

    def serialize_subgraph_to_text(self, subgraph: nx.MultiDiGraph) -> str:
        """Serializes the subgraph into a human-readable text format for the LVLM."""

        if subgraph.number_of_nodes() == 0:
            return "The scene appears to be empty or no relevant objects were found."

        # --- 步骤 1: 预处理，按帧组织所有信息 ---
        frames_data = defaultdict(lambda: {"nodes": {}, "relations": set()})

        # 收集所有帧的节点和空间关系
        for node_key, data in subgraph.nodes(data=True):
            frame_id = data.get("frame_id")
            track_id = data.get("track_id")
            if frame_id is not None and track_id is not None:
                frames_data[frame_id]["nodes"][track_id] = data

        for u, v, data in subgraph.edges(data=True):
            if data.get("type") == "spatial":
                frame_id = subgraph.nodes[u].get("frame_id")
                u_track_id = subgraph.nodes[u].get("track_id")
                v_track_id = subgraph.nodes[v].get("track_id")
                relation = data.get("relation")
                if all(x is not None for x in [frame_id, u_track_id, v_track_id, relation]):
                    # 将关系存储为元组以便于比较
                    frames_data[frame_id]["relations"].add((u_track_id, relation, v_track_id))

        if not frames_data:
            return "No valid frame data found in the subgraph."

        # 获取时间上排序的帧ID
        sorted_frame_ids = sorted(frames_data.keys())

        # --- 步骤 2: 逐帧生成文本描述 ---
        final_text_parts = []

        for i, frame_id in enumerate(sorted_frame_ids):
            current_frame_data = frames_data[frame_id]

            # 获取任意一个节点来读取时间戳信息
            any_node_data = next(iter(current_frame_data["nodes"].values()))
            capture_time = any_node_data.get('frame_id') / self.fps if self.fps > 0 else 0.0
            obs_ts = any_node_data.get("view_ts", capture_time)

            # -- 构建当前帧的描述 --
            frame_text_parts = []
            # 帧头信息
            frame_text_parts.append(
                f"--- Frame {frame_id} (captured at {capture_time:.2f}s, observed by user at {obs_ts:.2f}s) ---")

            # 节点信息
            node_descs = ["Scene Objects:"]
            for track_id, data in sorted(current_frame_data["nodes"].items()):
                class_name = data.get('class_name', 'object')
                centroid = [float(round(c, 3)) for c in data.get("centroid_3d", [])]
                size_3d = [float(round(s, 3)) for s in data.get("size_3d", [])]
                node_descs.append(f" - A {class_name} (ID {track_id}) is present at 3D centroid {centroid}, and 3D size of [{size_3d}].")
            frame_text_parts.append("\n".join(node_descs))

            # 空间关系信息
            if current_frame_data["relations"]:
                relation_descs = ["Spatial Relations:"]
                for subj_id, rel, obj_id in sorted(list(current_frame_data["relations"])):
                    subj_name = current_frame_data.get("nodes", {}).get(subj_id, {}).get("class_name", "object")
                    obj_name = current_frame_data.get("nodes", {}).get(obj_id, {}).get("class_name", "object")
                    relation_descs.append(f" - The {subj_name} (ID {subj_id}) is {rel} the {obj_name} (ID {obj_id}).")
                frame_text_parts.append("\n".join(relation_descs))

            # -- [核心] 与前一帧进行对比，生成变化描述 --
            if i > 0:
                prev_frame_id = sorted_frame_ids[i - 1]
                prev_frame_data = frames_data[prev_frame_id]

                current_track_ids = set(current_frame_data["nodes"].keys())
                prev_track_ids = set(prev_frame_data["nodes"].keys())

                diff_descs = []
                # 1. 消失的物体
                disappeared_ids = prev_track_ids - current_track_ids
                for track_id in sorted(list(disappeared_ids)):
                    class_name = prev_frame_data["nodes"][track_id].get("class_name", "object")
                    diff_descs.append(f" - Object Disappeared: The {class_name} (ID {track_id}) is no longer visible.")

                # 2. 出现的物体
                appeared_ids = current_track_ids - prev_track_ids
                for track_id in sorted(list(appeared_ids)):
                    class_name = current_frame_data["nodes"][track_id].get("class_name", "object")
                    diff_descs.append(f" - Object Appeared: A new {class_name} (ID {track_id}) has become visible.")

                # 3. 变化的空间关系
                current_relations = current_frame_data["relations"]
                prev_relations = prev_frame_data["relations"]

                added_relations = current_relations - prev_relations
                removed_relations = prev_relations - current_relations

                for subj_id, rel, obj_id in sorted(list(removed_relations)):
                    # 只报告那些主体和客体仍然存在的旧关系，这通常意味着关系本身发生了改变
                    if subj_id in current_track_ids and obj_id in current_track_ids:
                        subj_name = current_frame_data["nodes"][subj_id].get("class_name", "object")
                        obj_name = current_frame_data["nodes"][obj_id].get("class_name", "object")
                        diff_descs.append(
                            f" - Relation Ended: The {subj_name} (ID {subj_id}) is no longer {rel} the {obj_name} (ID {obj_id}).")

                for subj_id, rel, obj_id in sorted(list(added_relations)):
                    if subj_id in current_track_ids and obj_id in current_track_ids:
                        subj_name = current_frame_data["nodes"][subj_id].get("class_name", "object")
                        obj_name = current_frame_data["nodes"][obj_id].get("class_name", "object")
                        diff_descs.append(
                            f" - Relation Started: The {subj_name} (ID {subj_id}) is now {rel} the {obj_name} (ID {obj_id}).")

                if diff_descs:
                    frame_text_parts.append("Changes from previous frame:\n" + "\n".join(diff_descs))

            final_text_parts.append("\n\n".join(frame_text_parts))

            # --- 步骤 3: 组合所有文本并添加脚注 ---
        final_text = "\n\n".join(final_text_parts)

        final_text += "\n\n---\nCoordinate System Note: All 3D coordinates (centroid_3d) are in meters, relative to the camera's perspective (X-right, Y-down, Z-forward)."

        return final_text


        # text_parts = ["The current scene contains these relevant objects and relationships:"]
        #
        # scene_descript = {}
        # for node_key, data in subgraph.nodes(data=True):
        #     frame_id = data.get("frame_id")
        #     obs_ts = data.get("view_ts", 0.0)
        #     capture_time = data.get("capture_time", 0.0)
        #     if frame_id is None:
        #         continue
        #     frame_key = f"frame_{frame_id}"
        #     if frame_key not in scene_descript:
        #         head_descript = f"The {frame_key} is captured at timestamp {capture_time:.3f}, the user observed at timestamp {obs_ts:.3f}."
        #         scene_descript[frame_key] = {"head":[head_descript], "nodes": ["There is"], "edges": ["The relation between them is following:"],
        #                                      "diff":["The changes to the previous frame are:"]}
        #
        #     class_name = data.get('class_name', 'unknown object')
        #     track_id = data.get("track_id", -1)
        #     centroid_3d = [round(c, 4) for c in data.get("centroid_3d", [])]
        #     size_3d = [round(s, 4) for s in data.get("size_3d", [])]
        #     isntance_descript = f"a {class_name} with Id {track_id} (node '{node_key}'), 3D centroid at {centroid_3d}, and 3D size of {size_3d};"
        #     scene_descript[frame_key]["nodes"].append(isntance_descript)
        #
        # for u, v, data in subgraph.edges(data=True):
        #     u_data = subgraph.nodes[u]
        #     v_data = subgraph.nodes[v]
        #     u_desc = f"The {u_data.get('class_name')} (ID {u_data.get('track_id')})"
        #     v_desc = f"the {v_data.get('class_name')} (ID {v_data.get('track_id')})"
        #     relation_type = data.get("type")
        #     relation = data.get("relation")
        #     if relation_type == "spatial":
        #         edge_descriptions = f" - {u_desc} is {relation} {v_desc}."
        #         frame_key = f"frame_{u_data.get('frame_id')}"
        #         scene_descript[frame_key]["edges"].append(edge_descriptions)
        #     elif relation_type == "temporal" and u_data.get('frame_id') != v_data.get('frame_id'):
        #         diff_descriptions = f"xxxxxx is shown or xxxx is dispared"
        #         scene_descript[frame_key]["diff"].append(edge_descriptions)
        #
        #
        # if len(edge_descriptions) > 1:
        #     text_parts.extend(edge_descriptions)
        #
        # text_parts.append(
        #     "\nCoordinate System Note: All 3D coordinates (centroid_3d) are in meters, relative to the camera's perspective (X-right, Y-down, Z-forward).")
        # return "\n".join(text_parts)

    def generate_llm_prompt(self, user_command: str, com_ts: float = 0.0):
        """Assembles the final comprehensive prompt for the LVLM."""

        root_track_ids = self.find_relevant_entities(user_command, top_k=5)
        if not root_track_ids:
            print("无法将用户指令接地到场景中的任何物体。任务中止。")
            return {"error": "无法识别相关物体。"}

        subgraph = self.extract_task_subgraph(root_track_ids)

        world_state_json = self.serialize_subgraph_to_json(subgraph)
        world_state_text = self.serialize_subgraph_to_text(subgraph)

        full_prompt = self.build_latency_aware_planner_prompt(
            user_command=user_command,
            com_ts=str(com_ts),
            latency_s=self.latency,
            world_state_json=str(world_state_json),
            world_state_text=world_state_text # concise NL description of the same subgraph
        )

        return full_prompt
    def build_latency_aware_planner_prompt(
            self,
            user_command: str,
            com_ts: str,
            latency_s: float,
            world_state_json: str,
            # serialized JSON string of the subgraph (with per-frame timestamps, centroids, sizes)
            world_state_text: str  # concise NL description of the same subgraph
    ) -> str:

        #- World state JSON: {world_state_json} 这行被我删掉了，在CONTEXT VARIABLES的最后一行
        template = f"""SYSTEM:
    You are an LVLM planner for a fixed-position, single-arm robot. The robot cannot move its base and may only use the following primitive actions: Push, Pull, Slide, Press, Pick, Place.

    VIDEO INPUT:
    You will be given a video captured by the robot’s camera over a period of time. 
    Watch the video carefully before generating any output.

    SCENE GRAPH INPUT:
    I have constructed a video scene graph from these frames and extracted a subgraph relevant to the user command:
    - world_state_json: a structured subgraph of relevant nodes/edges (objects, attributes, 3D centroids, 3D sizes, relations, per-frame timestamps).
    - world_state_text: a natural-language description of the same subgraph for redundancy and disambiguation.
    Both describe the CURRENT scene the robot will act on, and also include per-frame timestamps (capture_ts: when the frame was recorded by the camera; observe_ts: when this frame was shown to the user) helpful for temporal reasoning.

    CONTEXT VARIABLES:
    - User command (natural language): {user_command}
    - Command absolute timestamp (UTC or epoch): {com_ts}
    - Known one-way latency from user to robot (seconds): {latency_s}
    - World state Text: {world_state_text}

    LATENCY MODEL & INTENTION RECONSTRUCTION (must do before planning):
    1) Reconstruct what the user likely SAW at {com_ts}:
       - Prefer frames whose frame.observe_ts is closest to {com_ts}. If observe_ts is unavailable, approximate using frames with capture_ts closest to {com_ts}-{latency_s}.
       - From those frames, world_state_json and world_state_text, derive S_user: the set of target candidates consistent with the command semantics.
    2) Infer the INTENDED INSTANCE at command time:
       - If multiple candidates in S_user, choose the best match by attributes and salience (visibility, proximity, stability).
    3) Align to CURRENT world state:
       - Map the intended instance to the current scene using world_state_json/text. Handle occlusion/duplication/move/rotation/removal with pre-steps or best-alternative selection.
       - Do NOT invent objects.

    PLANNING RULES:
    - Use ONLY: Push, Pull, Slide, Press, Pick, Place.
    - Every manipulation step MUST include the object's (centroid, size) from world_state_json/text:
      "<verb> <object_name> (centroid:[x y z], size:[sx sy sz])"
    - Reuse "it" for subsequent steps but keep centroid/size.
    - Keep the plan minimal and safe; add pre-steps if needed.

    OUTPUT FORMAT (strict):
    - Return ONLY a JSON array of strings (no prose, no extra keys).

    VALIDATION:
    - Verbs ∈ {{Push, Pull, Slide, Press, Pick, Place}}.
    - Each manipulation step includes centroid/size from world_state_json/text.
    - The selected instance must be consistent with the latency-aware intention at {com_ts} and available now.
    - If not feasible, return [].
    """
        return template


def generate_prompt(video_graph:VideoSceneGraphBuilder, user_prompt, com_ts):
    planner = TaskPlanner(vsg_builder=video_graph)
    prompt = planner.generate_llm_prompt(user_command=user_prompt, com_ts=com_ts)
    # print("prompt:", prompt)
    planner.graph2json("full_scene_graph.json")
    return prompt