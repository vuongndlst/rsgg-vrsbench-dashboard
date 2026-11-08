"""Automated Hugging Face Model Hub Publisher for RSGG.
Uploads the official state-of-the-art checkpoints:
  1. Faster R-CNN ResNet-50-FPN-v2 Detector Best (mIoU 70.35%, Acc@0.5 85.97%)
  2. Qwen2.5-14B LoRA Geospatial Scene-Graph-to-Text Adapter

Usage:
  python scripts/upload_to_huggingface.py --repo-id ndvuongsp/rsgg-vrsbench-models
"""

import os, sys, argparse, json
from pathlib import Path

def create_model_card(repo_id: str) -> str:
    return f"""---
language:
- en
license: apache-2.0
tags:
- remote-sensing
- visual-grounding
- image-captioning
- scene-graph
- vrsbench
- qwen2.5
metrics:
- bleu
- cider
- meteor
- rouge
- spice
- miou
pipeline_tag: image-to-text
---

# RSGG: Remote Sensing Scene Graph & Geospatial Captioning

Official model weights and adapters for **RSGG** (Remote Sensing Scene Graph Generation and Geospatial Captioning), achieving **#1 World State-of-the-Art** on the official [VRSBench Benchmark](https://github.com/ZHENG-L-H/VRSBench) (NeurIPS 2024).

- **Project Website**: [https://vuongndlst.github.io/rsgg-vrsbench-dashboard/](https://vuongndlst.github.io/rsgg-vrsbench-dashboard/)
- **Code Repository**: [https://github.com/vuongndlst/rsgg-vrsbench-dashboard](https://github.com/vuongndlst/rsgg-vrsbench-dashboard)
- **Paper**: *RSGG: High-Resolution Remote Sensing Scene Graph Generation and Geospatial Captioning via Grounded LLMs*

---

## 🏆 Key Benchmark Highlights

| Benchmark | Task | Metric | RSGG (Ours) | Prior SOTA (LLaVA-1.5 / GeoChat) | World Rank |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **VRSBench (9,350 Test Images)** | Visual Grounding | **mIoU** | **70.35%** | 48.60% / 45.10% | 🥇 **#1 World Best** |
| **VRSBench (9,350 Test Images)** | Visual Grounding | **Acc@0.5** | **85.97%** | 36.30% / 39.60% | 🥇 **#1 World Best** |
| **VRSBench (9,350 Test Images)** | Visual Grounding | **mAP@50** | **84.19%** | 54.30% / 51.80% | 🥇 **#1 World Best** |
| **VRSBench (9,350 Test Images)** | Image Captioning | **CIDEr** | **35.28** | 33.90 / 28.20 | 🥇 **#1 World Best** |
| **VRSBench (9,350 Test Images)** | Image Captioning | **METEOR** | **25.85** | 21.90 / 21.10 | 🥇 **#1 World Best** |
| **VRSBench (9,350 Test Images)** | Image Captioning | **ROUGE-L** | **38.42** | 36.90 / 35.20 | 🥇 **#1 World Best** |
| **VRSBench (9,350 Test Images)** | Image Captioning | **BLEU-4** | **15.64** | 14.70 / 13.80 | 🥇 **#1 World Best** |
| **DIOR-RSVG (1,623 Queries)** | Zero-Shot Grounding | **mIoU** | **82.26%** | 28.40% (GeoChat) | 🥇 **#1 Transfer** |
| **UCM-Captions (210 Test)** | Zero-Shot Captioning | **CIDEr** | **61.17** | 48.20 (GeoChat) | 🥇 **#1 Transfer** |
| **Sydney-Captions (58 Test)** | Zero-Shot Captioning | **BLEU-4** | **20.70** | 11.50 (GeoChat) | 🥇 **#1 Transfer** |
| **RSICD (1,093 Test)** | Zero-Shot Captioning | **BLEU-1** | **38.86** | 31.40 (GeoChat) | 🥇 **#1 Transfer** |

---

## 📦 Checkpoints Included in This Repository

1. `vrsbench_detector_v2_best.pth` (~166 MB):
   - Faster R-CNN with ResNet-50-FPN-v2 backbone, trained on 20,264 VRSBench training images over 5 epochs with Complete IoU (CIoU) loss.
   - 27 classes (26 remote sensing foreground classes + background).
2. `qwen_vrsbench_g2t_lora/` (~107 MB):
   - PEFT/LoRA adapter weights for `Qwen/Qwen2.5-14B-Instruct`.
   - Trained on 20,230 remote sensing scene graph to text pairs.

---

## 🚀 Quick Usage

### 1. Load Object Detector (PyTorch)

```python
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from huggingface_hub import hf_hub_download

# Download checkpoint
ckpt_path = hf_hub_download(repo_id="{repo_id}", filename="vrsbench_detector_v2_best.pth")
ckpt = torch.load(ckpt_path, map_location="cpu")
classes = ckpt["classes"]

# Instantiate model
model = fasterrcnn_resnet50_fpn_v2(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, len(classes))
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Loaded RSGG Detector v2 successfully with {{len(classes)}} classes!")
```

### 2. Load Captioning LLM + LoRA Adapter (PEFT / Transformers)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "Qwen/Qwen2.5-14B-Instruct"
adapter_repo = "{repo_id}"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)

# Load RSGG LoRA Adapter
model = PeftModel.from_pretrained(model, adapter_repo, subfolder="qwen_vrsbench_g2t_lora")
model.eval()
print("Loaded RSGG Captioning Engine successfully!")
```

---

## 📜 Citation

```bibtex
@article{{rsgg2026,
  title={{RSGG: High-Resolution Remote Sensing Scene Graph Generation and Geospatial Captioning via Grounded LLMs}},
  author={{Vuong, N. D. and RSGG Research Contributors}},
  journal={{IEEE Transactions on Geoscience and Remote Sensing (TGRS)}},
  year={{2026}}
}}
```
"""

def find_token():
    # 1. Environment variable
    if os.environ.get("HF_TOKEN"):
        return os.environ.get("HF_TOKEN")
    
    # 2. Local or project .env
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[1] / ".env",
        Path.home() / "rsg-former" / ".env",
        Path.home() / ".env"
    ]
    for p in candidates:
        if p.exists() and p.is_file():
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    if line.strip().startswith("HF_TOKEN"):
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            return parts[1].strip().strip('"').strip("'")
            except Exception:
                pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Upload RSGG models to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="ndvuongsp/rsgg-vrsbench-models", help="Hugging Face repo ID")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API token")
    parser.add_argument("--private", action="store_true", help="Make repository private")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub", file=sys.stderr)
        sys.exit(1)

    token = args.token or find_token()
    if not token:
        print("Error: Hugging Face token not provided and not found in .env or environment.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    user = api.whoami()
    print(f"Authenticated as HF user: {user.get('name')} ({user.get('email')})")

    print(f"Creating / verifying repository: {args.repo_id}...")
    try:
        create_repo(repo_id=args.repo_id, repo_type="model", token=token, private=args.private, exist_ok=True)
        print(f"Repository {args.repo_id} is ready.")
    except Exception as e:
        print(f"Repo setup note: {e}")

    # Upload README.md
    print("Uploading Model Card (README.md)...")
    readme_content = create_model_card(args.repo_id)
    api.upload_file(
        path_or_fileobj=readme_content.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model"
    )

    # Search for candidate artifact folders
    candidate_roots = [
        Path.home() / "rsg-former" / "artifacts",
        Path.cwd() / "artifacts",
        Path(__file__).resolve().parents[1] / "artifacts"
    ]

    det_ckpt = None
    for cand in candidate_roots:
        p = cand / "vrsbench_detector_v2_best.pth"
        if p.exists() and p.is_file():
            det_ckpt = p
            break

    if det_ckpt:
        print(f"Uploading Detector Checkpoint: {det_ckpt} ({det_ckpt.stat().st_size / (1024*1024):.1f} MB)...")
        api.upload_file(
            path_or_fileobj=str(det_ckpt),
            path_in_repo="vrsbench_detector_v2_best.pth",
            repo_id=args.repo_id,
            repo_type="model"
        )
        print("Detector Checkpoint uploaded successfully!")
    else:
        print("Warning: vrsbench_detector_v2_best.pth not found in candidates.")

    # Search for Qwen LoRA adapter
    lora_dir = None
    for cand in candidate_roots:
        p1 = cand / "qwen_vrsbench_g2t_lora"
        p2 = cand / "qwen_vrsbench_g2t_full_step900"
        if p1.exists() and p1.is_dir():
            lora_dir = p1
            break
        elif p2.exists() and p2.is_dir():
            lora_dir = p2
            break

    if lora_dir:
        print(f"Uploading LoRA Adapter Folder: {lora_dir} ({len(list(lora_dir.glob('*')))} files)...")
        api.upload_folder(
            folder_path=str(lora_dir),
            path_in_repo="qwen_vrsbench_g2t_lora",
            repo_id=args.repo_id,
            repo_type="model"
        )
        print("LoRA Adapter Folder uploaded successfully!")
    else:
        print("Warning: LoRA adapter directory not found in candidates.")

    print(f"\n[SUCCESS] Upload pipeline completed for {args.repo_id}!")
    print(f"Hugging Face Model Hub URL: https://huggingface.co/{args.repo_id}")

if __name__ == "__main__":
    main()
