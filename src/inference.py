"""
Inference generation routines with Qwen2.5-VL and QLoRA.
"""

import os
import gc
import json
from typing import Dict, Any, List

os.environ["FORCE_QWENVL_VIDEO_READER"] = "decord"

def get_processor(model_id: str):
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(model_id)

def build_messages(video_path: str, prompt: str, nframes: int = 24, resized: int = 224):
    return [{
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": video_path,
                "nframes": nframes,
                "resized_height": resized,
                "resized_width": resized
            },
            {"type": "text", "text": prompt}
        ]
    }]

def prepare_inputs(processor, video_path: str, prompt: str, nframes: int = 24, resized: int = 224):
    from qwen_vl_utils import process_vision_info
    msgs = build_messages(video_path, prompt, nframes=nframes, resized=resized)
    imgs, vids, vk = process_vision_info(msgs, return_video_kwargs=True)
    if isinstance(vk, dict):
        fps = vk.get("fps")
        if isinstance(fps, (list, tuple)) and len(fps) == 1:
            vk["fps"] = float(fps[0])
    text = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return processor(
        text=[text], images=imgs, videos=vids, padding=True, return_tensors="pt",
        **(vk if isinstance(vk, dict) else {})
    )

def generate_prediction(model, processor, video_path: str, prompt: str, max_new_tokens: int = 256):
    import torch
    with torch.inference_mode():
        inp = prepare_inputs(processor, video_path, prompt)
        inp = {k: (v.to(model.device) if torch.is_tensor(v) else v) for k, v in inp.items()}
        n = inp["input_ids"].shape[1]
        ids = model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode([o[n:] for o in ids], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    del inp, ids
    gc.collect()
    torch.cuda.empty_cache()
    return text

def load_qwen_model(model_id: str, adapter_path: str = None):
    """Loads 4-bit quantized Qwen2.5-VL with optional LoRA adapter."""
    import torch
    from transformers import Qwen2_5_VLForConditionalGeneration, BitsAndBytesConfig
    
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16
    )
    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb,
        device_map="auto"
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, adapter_path)
        return model
    return base
