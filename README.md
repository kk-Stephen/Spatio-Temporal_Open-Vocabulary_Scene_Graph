# VSG-Planner: Spatial-Temporal Video Scene Graph for Latency-Aware Robot Planning

This repository contains the official implementation of **VSG-Planner**, a framework that constructs spatial-temporal video scene graphs from RGB-D streams and uses them to generate structured prompts for VLM-based robot action planning under communication latency.

## Overview

In teleoperation and cloud-robot scenarios, communication latency causes the user's observation to lag behind the actual scene state. VSG-Planner addresses this by:

1. **Building a Video Scene Graph (VSG)** from RGB-D frames — tracking objects across time with 3D geometry, CLIP embeddings, and spatial relations.
2. **Extracting task-relevant subgraphs** grounded to the user's natural language command.
3. **Generating latency-aware prompts** that enable a VLM planner to reconstruct the user's intent at command time and align it to the current scene state.

### Pipeline

```
RGB-D Video  ──►  Preprocessing  ──►  Frame-by-Frame Detection  ──►  Video Scene Graph  ──►  Task Planner  ──►  Robot Action Plan
                  (preprocessing.py)   (LVLM + Grounded DINO)       (scene_graph.py)        (planner.py)        (VLM inference)
```

## Repository Structure

```
VSG-Planner/
│
├── scene_graph.py            # [CORE] Video Scene Graph builder — object tracking,
│                             #   3D reconstruction, spatial relation inference
├── planner.py                # [CORE] Task planner — subgraph extraction, prompt generation
├── preprocessing.py          # Extract RGB-D frames from RealSense .bag files
│
├── run_single.py             # Run pipeline on a single video
├── run_pipeline.py           # Batch pipeline over a video dataset (with labels)
├── run_pipeline_latency.py   # Batch pipeline with latency simulation (from CSV)
│
├── inference.py              # Standalone VLM inference (load prompt + video → plan)
├── inference_latency.py      # Latency-aware VLM inference variant
├── inference_batch.py        # Batch VLM inference over a dataset
├── evaluate.py               # Evaluation with Hungarian-matching semantic similarity
│
├── models/
│   ├── qwen25_vl.py          # Qwen2.5-VL model wrapper (3B/7B/32B)
│   └── grounded_dino.py      # Grounded DINO + SAM2 wrapper (detection & segmentation)
│
├── configs/
│   └── grounded_dino.yaml    # Model paths and thresholds for Grounded DINO
│
├── prompts/
│   ├── frame_detect_prompt.txt   # Prompt template for per-frame object & relation detection
│   ├── extra.txt                 # Additional instructions appended to planner prompts
│   ├── video_descript_prompt.txt  # Video description prompt
│   └── frame_eval_prompt.txt     # Frame evaluation prompt
│
├── util/
│   └── utils.py              # Model loading, config parsing, file path utilities
│
├── datasets/
│   ├── latencyplan_qwen.py   # PyTorch Dataset class (WIP)
│   └── frame_cut.py          # Extract frames from video files at target FPS
│
├── tools/                    # Auxiliary utility scripts
│   ├── compress.py           # Batch video compression (ffmpeg)
│   ├── count.py              # Count large prompt files
│   ├── count_edge.py         # Analyze edge statistics in saved scene graph JSONs
│   ├── show_pcd.py           # Visualize 3D meshes with Open3D
│   └── eval_moverscore.py    # MoverScore evaluation metric
│
├── Grounded_SAM_2_main/      # [EXTERNAL] Grounded SAM 2 — replaceable with any
│                             #   open-vocabulary object detector
├── weights/                  # Pre-trained model weights (Qwen2.5-VL-3B/7B)
├── requirements.txt
└── README.md
```

## Installation

### 1. Clone and set up the environment

```bash
git clone https://github.com/<your-username>/VSG-Planner.git
cd VSG-Planner

conda create -n vsg-planner python=3.10 -y
conda activate vsg-planner

pip install -r requirements.txt
```

### 2. Install Grounded SAM 2 (or your preferred open-vocabulary detector)

The `Grounded_SAM_2_main/` directory contains a copy of [Grounded SAM 2](https://github.com/IDEA-Research/Grounded-SAM-2). Follow its installation instructions to set up the model checkpoints:

```bash
cd Grounded_SAM_2_main
# Download SAM2 and Grounding DINO checkpoints per their README
# Expected paths (configurable in configs/grounded_dino.yaml):
#   checkpoints/sam2.1_hiera_large.pt
#   gdino_checkpoints/groundingdino_swinb_cogcoor.pth
cd ..
```

> **Note:** This module is replaceable. Any detector that returns bounding boxes and segmentation masks in COCO RLE format can be substituted.

### 3. Download model weights

- **Qwen2.5-VL-3B** (scene description): Place in `weights/qwen25_3B/`
- **Qwen2.5-VL-7B** (optional): Place in `weights/qwen25_7B/`
- **Qwen2.5-VL-32B** (planning): Used via HuggingFace model name or local path
- **CLIP** (`openai/clip-vit-base-patch32`): Downloaded automatically on first run

## Usage

### Preprocessing: Extract RGB-D frames from a RealSense .bag file

```bash
python preprocessing.py
# Edit INPUT_BAG, OUTPUT_DIR, FPS_TARGET at the top of the file
```

Output: paired `{idx}_rgb.png` and `{idx}_depth.png` files.

### Single video pipeline

Build a scene graph and generate a planning prompt for one video:

```bash
python run_single.py \
  --video_path path/to/video.mp4 \
  --frames_dir path/to/extracted_frames/ \
  --fps 2 \
  --latency 1.5 \
  --prompt "Pick up the red cup" \
  --com_ts 8.0 \
  --save_prompt_dir output_prompt.txt
```

### Batch pipeline

Process multiple videos with label files:

```bash
python run_pipeline.py \
  --video_dir path/to/videos/ \
  --label_dir path/to/labels/ \
  --output_prompt_dir full_prompts/ \
  --output_csv_path planning_dataset.csv \
  --fps 2 \
  --latency 0.0
```

### Batch pipeline with latency simulation

```bash
python run_pipeline_latency.py \
  --video_dir path/to/videos/ \
  --csv_path input_tasks.csv \
  --output_prompt_dir full_prompts/ \
  --output_csv_path results.csv
```

The input CSV should contain columns: `video_name`, `instruction`, `com_ts`, `latency`.

### Inference (generate robot plan from a pre-built prompt)

```bash
python inference.py \
  --video_path path/to/video.mp4 \
  --prompt_dir full_prompts/video_0_inst_0.txt
```

### Evaluation

```bash
python evaluate.py \
  --csv_path planning_dataset.csv \
  --video_dir path/to/videos/ \
  --output_csv_path evaluation_results.csv
```

Uses sentence embedding similarity (all-MiniLM-L6-v2) with Hungarian matching to compare predicted action plans against ground truth.

## Core Components

### Video Scene Graph Builder (`scene_graph.py`)

The central contribution. For each RGB-D frame:

1. **LVLM Detection** — Qwen2.5-VL identifies objects and spatial relations from the RGB image.
2. **Grounded Segmentation** — Grounded DINO + SAM2 produces bounding boxes and instance masks.
3. **3D Reconstruction** — Open3D creates per-object point clouds from aligned RGB-D data, extracting 3D centroids and oriented bounding box sizes.
4. **CLIP Embeddings** — Visual and semantic embeddings for each detected instance.
5. **Cross-Frame Tracking** — Hungarian algorithm matches instances across frames using spatial distance (60%) and visual similarity (40%), with a class name consistency penalty.
6. **Graph Construction** — A NetworkX `MultiDiGraph` with:
   - **Nodes**: Object instances at specific frames (keyed as `{track_id}@{frame_id}`)
   - **Temporal edges**: Connect the same object across consecutive frames
   - **Spatial edges**: Encode intra-frame relations (e.g., "on", "next to", "holding")

### Task Planner (`planner.py`)

Given a user command and the built scene graph:

1. **Entity Grounding** — Finds the top-K most relevant tracked objects using CLIP cosine similarity between the command text and object embeddings.
2. **Subgraph Extraction** — Extracts the task-relevant subgraph (target objects + their spatial neighbors across all frames).
3. **Prompt Assembly** — Generates a structured prompt with latency-aware reasoning instructions, 3D object properties, and temporal context for the VLM planner.

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_SPATIAL_DIST` | 1.5 m | Max 3D distance for cross-frame object matching |
| `MATCHING_COST_THRESHOLD` | 0.7 | Cost threshold for track association |
| `w_spatial` | 0.6 | Weight for spatial features in matching |
| `w_visual` | 0.4 | Weight for visual embeddings in matching |
| `BOX_THRESHOLD` | 0.35 | Grounding DINO detection confidence |
| `TEXT_THRESHOLD` | 0.25 | Grounding DINO text alignment threshold |
| Camera intrinsics | 640x480, fx=517.3, fy=516.5 | Default RealSense D435 parameters |
| `depth_scale` | 5000.0 | Depth image scale factor |
| `depth_trunc` | 5.0 m | Maximum depth truncation |

## Data Format

### Input frames
- RGB: `{idx}_rgb.png` (8-bit, 3-channel)
- Depth: `{idx}_depth.png` (16-bit, single-channel, millimeters)

### Label JSON format
```json
[
  {
    "instruction": "Pick up the cup and place it next to the keyboard",
    "action": ["Pick cup", "Place cup next to keyboard"]
  }
]
```

### Scene Graph JSON output
```json
{
  "metadata": { "total_frames": 10, "total_tracks": 5, "fps": 2, "latency": 1.5 },
  "frame_states": {
    "frame_0": {
      "nodes": [
        { "node_id": "0@0", "track_id": 0, "class_name": "cup",
          "centroid_3d": [0.12, -0.03, 0.45], "size_3d": [0.08, 0.08, 0.12] }
      ],
      "edges": [
        { "source": "0@0", "target": "1@0", "type": "spatial", "relation": "on" }
      ]
    }
  },
  "temporal_edges": [
    { "source": "0@0", "target": "0@1", "type": "temporal", "relation": "next_frame" }
  ]
}
```

## Citation

This work has been accepted by **ICRA 2026** (IEEE International Conference on Robotics and Automation, June 2026). If you find this work useful, please cite:

```bibtex
@inproceedings{wang2025open,
  title={Open-Vocabulary Spatio-Temporal Scene Graph for Robot Perception and Teleoperation Planning},
  author={Wang, Yi and Xue, Zeyu and Liu, Mujie and Zhang, Tongqin and Hu, Yan and Zhao, Zhou and Yang, Chenguang and Lu, Zhenyu},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2026}
}
```

arXiv preprint: [arXiv:2509.23107](https://arxiv.org/abs/2509.23107)

## License

TODO: Add license information.
