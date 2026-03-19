import torch
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

class LVLM(nn.Module):
    def __init__(self, model_name: str, dtype: torch.dtype = torch.bfloat16, device_map: str = "auto"):
        super().__init__()
        if model_name == "Qwen/Qwen2.5-VL-3B-Instruct":
            model_pth = "./weights/qwen25_3B"
        elif model_name == "Qwen/Qwen2.5-VL-7B-Instruct":
            model_pth = "./weights/qwen25_7B"
        else:
            model_pth = model_name
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_pth,
            torch_dtype=dtype,
            device_map=device_map
        )
        self.processor = AutoProcessor.from_pretrained(model_pth)

    def prepare_inputs(self, messages):
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        )
        return inputs.to(self.model.device)

    def generate(self, messages, max_new_tokens=2048):
        inputs = self.prepare_inputs(messages)
        generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        trimmed_ids = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output = self.processor.batch_decode(
            trimmed_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output


class SetCriterion(nn.Module):
    def __init__(self):
        super().__init__()
        # 暂时空实现



def build_model(args, device):
    if args.dtype == "float32":
        dtype = torch.float32
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = "auto"

    model = LVLM(args.model_name, dtype=dtype)
    criterion = SetCriterion()
    return model, criterion