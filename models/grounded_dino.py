import torch
import numpy as np
import pycocotools.mask as mask_util
from pathlib import Path
from torchvision.ops import box_convert
from torch import nn
from Grounded_SAM_2_main.sam2.build_sam import build_sam2
from Grounded_SAM_2_main.sam2.sam2_image_predictor import SAM2ImagePredictor
from Grounded_SAM_2_main.grounding_dino.groundingdino.util.inference import load_model, load_image, predict
# from sam2.build_sam import build_sam2
# from sam2.sam2_image_predictor import SAM2ImagePredictor
# from groundingdino.util.inference import load_model, load_image, predict
from contextlib import nullcontext

class GroundedDINO(nn.Module):
    def __init__(
        self,
        sam2_cfg: str,
        sam2_ckpt: str,
        dino_cfg: str,
        dino_ckpt: str,
        box_threshold: float,
        text_threshold: float,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        # Build and prepare SAM2 model
        sam2_model = build_sam2(sam2_cfg, sam2_ckpt, device=self.device)
        self.sam2_predictor = SAM2ImagePredictor(sam2_model)

        # Build Grounding DINO model
        self.grounding_model = load_model(
            model_config_path=dino_cfg,
            model_checkpoint_path=dino_ckpt,
            device=self.device
        )

        # Move module to target device
        # self.to(self.device)

    @torch.no_grad()
    def forward(self, prompt: str, image_path: str, dump_json: bool = True) -> dict:
        # Load and set up image for both models
        image_source, image_input = load_image(image_path)
        self.sam2_predictor.set_image(image_source)

        # Predict bounding boxes with grounding DINO
        # boxes, confidences, class_names = predict(
        #     model=self.grounding_model,
        #     image=image_input,
        #     caption=prompt,
        #     box_threshold=self.box_threshold,
        #     text_threshold=self.text_threshold,
        #     device=self.device
        # )

        device_type = self.device.type if isinstance(self.device, torch.device) else str(self.device)
        with torch.cuda.amp.autocast(enabled=False) if device_type == "cuda" else nullcontext():
            # 确保输入是 float32，防止 dtype 传染
            image_input = image_input.to(self.device, dtype=torch.float32)
            boxes, confidences, class_names = predict(
                model=self.grounding_model,
                image=image_input,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )

        # Convert normalized cxcywh to pixel xyxy
        h, w, _ = image_source.shape
        boxes = boxes.to(self.device) * torch.tensor([w, h, w, h], device=self.device)
        xyxy = box_convert(boxes, in_fmt='cxcywh', out_fmt='xyxy').cpu().numpy()

        if xyxy.shape[0] == 0:
            return {
                'image_path': image_path,
                'annotations': [],
                'box_format': 'xyxy',
                'img_width': w,
                'img_height': h,
            }

        # Mixed precision context for SAM2 mask prediction
        # with torch.cuda.amp.autocast(device_type=self.device, dtype=torch.bfloat16):
        #     masks, scores, _ = self.sam2_predictor.predict(
        #         point_coords=None,
        #         point_labels=None,
        #         box=xyxy,
        #         multimask_output=False,
        #     )
        if isinstance(self.device, torch.device):
            device_type = self.device.type
        else:
            device_type = str(self.device)

        # torch.autocast(device_type=device_type, dtype=torch.bfloat16).__enter__()
        if device_type == "cuda" and torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        amp_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if device_type == "cuda" else nullcontext()
        with amp_ctx:
            masks, scores, logits = self.sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=xyxy,
                multimask_output=False,
            )
        # Squeeze if multimask dimension present
        if masks.ndim == 4:
            masks = masks.squeeze(1)

        confidences = confidences.tolist()
        annotations = []

        # Helper to convert mask to COCO RLE
        def mask_to_rle(mask: np.ndarray) -> dict:
            rle = mask_util.encode(
                np.array(mask[:, :, None], order='F', dtype='uint8')
            )[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle

        # Assemble annotation entries
        for cls_name, box, mask, score in zip(class_names, xyxy, masks, scores.tolist()):
            annotations.append({
                'class_name': cls_name,
                'bbox': box.tolist(),
                'segmentation': mask_to_rle(mask),
                'score': score,
            })

        result = {
            'image_path': image_path,
            'annotations': annotations,
            'box_format': 'xyxy',
            'img_width': w,
            'img_height': h,
        }

        return result

class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        # TODO: Implement matching and loss computation
        # 暂时空实现

def build_model(cfg, device: str = 'cpu'):
    cfg_model = cfg.MODEL
    model = GroundedDINO(
        sam2_cfg=cfg_model.SAM2_MODEL_CONFIG,
        sam2_ckpt=cfg_model.SAM2_CHECKPOINT,
        dino_cfg=cfg_model.GROUNDING_DINO_CONFIG,
        dino_ckpt=cfg_model.GROUNDING_DINO_CHECKPOINT,
        box_threshold=cfg_model.BOX_THRESHOLD,
        text_threshold=cfg_model.TEXT_THRESHOLD,
        device=device
    )
    criterion = SetCriterion()
    return model, criterion
