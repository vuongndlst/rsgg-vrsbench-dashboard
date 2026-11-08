"""Universal Caption Generation Engine for Remote Sensing.
Integrates:
  - Dynamic dataset prompt profiles (from configs/dataset_prompts.py)
  - Scene graph & spatial relationship extraction from bounding boxes
  - Beam Search with length penalty and n-gram repetition constraints
  - Zero-shot / In-domain adaptation without weight modification
"""

import os, sys, re, json
from typing import List, Dict, Any, Optional
from pathlib import Path
import torch

ROOT = Path.home() / 'rsg-former'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from configs.dataset_prompts import get_dataset_config

class RSGGCaptionEngine:
    def __init__(self, model_path: Optional[str] = None, device: str = "cuda:0"):
        self.device = device
        self.model = None
        self.tokenizer = None
        self.model_path = model_path
        self._load_model()

    def _load_model(self):
        """Lazy load Qwen2.5-14B-Instruct or lightweight fallback."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Priority search paths for local Qwen checkpoints
        candidates = [
            self.model_path,
            os.environ.get("QWEN_MODEL_DIR"),
            str(Path.home() / "etechs" / "msvd-tkg-captioning" / "models" / "qwen2.5-14b"),
            str(Path.home() / "shared-assets" / "models" / "Qwen2.5-14B-Instruct"),
            str(Path.home() / "models" / "Qwen2.5-14B-Instruct"),
            "Qwen/Qwen2.5-14B-Instruct"
        ]
        
        chosen_path = None
        for c in candidates:
            if c and (Path(c).exists() or not c.startswith("/")):
                chosen_path = c
                break
                
        device_target = self.device if self.device.startswith("cuda") else "cuda:0"
        print(f"[Engine] Loading LLM from: {chosen_path} with 4-bit quantization on {device_target}...")
        self.tokenizer = AutoTokenizer.from_pretrained(chosen_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load in 4-bit NormalFloat4 with bitsandbytes for ultra memory efficiency (~8.5GB VRAM)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            chosen_path,
            device_map={"": device_target},
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        self.model.eval()
        print("[Engine] Model loaded successfully.")

    @staticmethod
    def extract_spatial_scene_graph(detections: List[Dict[str, Any]], box_conf_threshold: float = 0.25, max_boxes: int = 12) -> str:
        """Convert bounding boxes and categories into a compact spatial scene description."""
        valid_boxes = [d for d in detections if d.get("confidence", 1.0) >= box_conf_threshold]
        valid_boxes = sorted(valid_boxes, key=lambda x: x.get("confidence", 1.0), reverse=True)[:max_boxes]
        
        if not valid_boxes:
            return "No prominent foreground objects detected (open landscape or general terrain)."

        # Group by category and assign spatial positions (top-left, center, bottom-right, etc.)
        cat_locations = {}
        for d in valid_boxes:
            cat = d.get("category", "object").lower()
            box = d.get("box", [0, 0, 1, 1])  # [ymin, xmin, ymax, xmax] in normalized coords
            cy = (box[0] + box[2]) / 2.0
            cx = (box[1] + box[3]) / 2.0
            
            # Rough 3x3 spatial grid
            v_pos = "top" if cy < 0.35 else ("bottom" if cy > 0.65 else "center")
            h_pos = "left" if cx < 0.35 else ("right" if cx > 0.65 else "center")
            pos = f"{v_pos}-{h_pos}" if f"{v_pos}-{h_pos}" != "center-center" else "center"
            
            if cat not in cat_locations:
                cat_locations[cat] = []
            cat_locations[cat].append(pos)

        desc_parts = []
        for cat, poses in cat_locations.items():
            count = len(poses)
            pos_summary = ", ".join(poses[:3])
            if count == 1:
                desc_parts.append(f"one {cat} at ({pos_summary})")
            else:
                desc_parts.append(f"{count} {cat}s around ({pos_summary})")

        return "; ".join(desc_parts)

    def build_prompt(self, dataset_name: str, scene_graph_text: str, custom_few_shot: Optional[List[Dict]] = None) -> str:
        """Construct the prompt using dataset-specific registry profiles."""
        cfg = get_dataset_config(dataset_name)
        sys_msg = cfg.get("system_instruction", "")
        exemplars = custom_few_shot if custom_few_shot is not None else cfg.get("few_shot_exemplars", [])

        prompt = f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        
        # Add few-shot exemplars
        for ex in exemplars:
            prompt += f"<|im_start|>user\nDetected objects: {ex['detected_objects']}\nGenerate caption:<|im_end|>\n"
            prompt += f"<|im_start|>assistant\n{ex['caption']}<|im_end|>\n"

        # Add target query
        prompt += f"<|im_start|>user\nDetected objects: {scene_graph_text}\nGenerate caption:<|im_end|>\n<|im_start|>assistant\n"
        return prompt

    @torch.inference_mode()
    def generate_caption(
        self,
        dataset_name: str,
        detections: List[Dict[str, Any]],
        beam_size: Optional[int] = None,
        length_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        box_conf_threshold: Optional[float] = None
    ) -> str:
        """Generate caption tailored to the specified dataset."""
        cfg = get_dataset_config(dataset_name)
        
        # Resolve hyperparameters
        k_beam = beam_size if beam_size is not None else cfg.get("beam_size", 3)
        l_pen = length_penalty if length_penalty is not None else cfg.get("length_penalty", 0.90)
        max_tok = max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 48)
        min_tok = cfg.get("min_new_tokens", 12)
        no_rep = cfg.get("no_repeat_ngram_size", 3)
        conf_thr = box_conf_threshold if box_conf_threshold is not None else cfg.get("box_confidence_threshold", 0.25)
        max_boxes = cfg.get("max_prompt_boxes", 12)

        # 1. Extract scene graph
        sg_text = self.extract_spatial_scene_graph(detections, box_conf_threshold=conf_thr, max_boxes=max_boxes)
        
        # 2. Build prompt
        prompt = self.build_prompt(dataset_name, sg_text)
        
        # 3. Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        input_len = inputs.input_ids.shape[1]

        # 4. Generate with Beam Search
        gen_kwargs = {
            "max_new_tokens": max_tok,
            "min_new_tokens": min_tok,
            "no_repeat_ngram_size": no_rep,
            "early_stopping": True,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id
        }
        if k_beam > 1:
            gen_kwargs["num_beams"] = k_beam
            gen_kwargs["length_penalty"] = l_pen
        else:
            gen_kwargs["do_sample"] = False  # greedy

        outputs = self.model.generate(**inputs, **gen_kwargs)
        gen_tokens = outputs[0][input_len:]
        caption = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Clean formatting
        caption = caption.replace("\n", " ").strip()
        if not caption.endswith("."):
            caption += "."
        return caption

    @torch.inference_mode()
    def generate_captions_batch(
        self,
        dataset_name: str,
        detections_list: List[List[Dict[str, Any]]],
        batch_size: int = 32,
        beam_size: Optional[int] = None,
        length_penalty: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
        box_conf_threshold: Optional[float] = None
    ) -> List[str]:
        """Generate captions for a batch of detection sets efficiently."""
        cfg = get_dataset_config(dataset_name)
        k_beam = beam_size if beam_size is not None else cfg.get("beam_size", 3)
        l_pen = length_penalty if length_penalty is not None else cfg.get("length_penalty", 0.90)
        max_tok = max_new_tokens if max_new_tokens is not None else cfg.get("max_new_tokens", 48)
        min_tok = cfg.get("min_new_tokens", 12)
        no_rep = cfg.get("no_repeat_ngram_size", 3)
        conf_thr = box_conf_threshold if box_conf_threshold is not None else cfg.get("box_confidence_threshold", 0.25)
        max_boxes = cfg.get("max_prompt_boxes", 12)

        self.tokenizer.padding_side = "left"
        all_captions = []

        for i in range(0, len(detections_list), batch_size):
            batch_dets = detections_list[i:i + batch_size]
            batch_prompts = []
            for dets in batch_dets:
                sg_text = self.extract_spatial_scene_graph(dets, box_conf_threshold=conf_thr, max_boxes=max_boxes)
                prompt = self.build_prompt(dataset_name, sg_text)
                batch_prompts.append(prompt)

            inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.model.device)
            gen_kwargs = {
                "max_new_tokens": max_tok,
                "min_new_tokens": min_tok,
                "no_repeat_ngram_size": no_rep,
                "early_stopping": True,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id
            }
            if k_beam > 1:
                gen_kwargs["num_beams"] = k_beam
                gen_kwargs["length_penalty"] = l_pen
            else:
                gen_kwargs["do_sample"] = False

            outputs = self.model.generate(**inputs, **gen_kwargs)
            input_len = inputs.input_ids.shape[1]
            for out in outputs:
                gen_tokens = out[input_len:]
                cap = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
                cap = cap.replace("\n", " ").strip()
                if not cap.endswith("."):
                    cap += "."
                all_captions.append(cap)

            cur_count = min(i + batch_size, len(detections_list))
            if (i // batch_size + 1) % 10 == 0 or cur_count == len(detections_list):
                print(f"  [Caption Batch] Completed {cur_count}/{len(detections_list)} captions...", flush=True)

        return all_captions

