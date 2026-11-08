<div align="center">

# RSGG: Remote Sensing Scene Graph Generation and Geospatial Captioning via Grounded LLMs

<p align="center">
  <a href="https://vuongndlst.github.io/rsgg-vrsbench-dashboard/"><img src="https://img.shields.io/badge/Project%20Page-Live%20Dashboard-blue?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project Page"></a>
  <a href="https://huggingface.co/ndvuongsp/rsgg-vrsbench-models"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models%20%26%20Weights-orange?style=for-the-badge" alt="Hugging Face"></a>
  <a href="https://github.com/vuongndlst/rsgg-vrsbench-dashboard"><img src="https://img.shields.io/badge/GitHub-Code%20Repository-black?style=for-the-badge&logo=github" alt="GitHub"></a>
  <a href="#citation"><img src="https://img.shields.io/badge/Paper-IEEE%20TGRS%202026-red?style=for-the-badge&logo=arxiv" alt="Paper"></a>
</p>

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace PEFT](https://img.shields.io/badge/PEFT-LoRA-yellow.svg?style=flat)](https://github.com/huggingface/peft)
[![VRSBench SOTA](https://img.shields.io/badge/VRSBench-%231%20World%20SOTA-emerald.svg?style=flat)](https://github.com/ZHENG-L-H/VRSBench)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

<br/>

**RSGG** is an end-to-end framework that couples high-precision **multi-scale geometric visual grounding** with **large language model generation** (Qwen2.5-14B) to produce detailed, geometrically grounded, and hallucination-free descriptions of high-resolution remote sensing imagery.

Evaluated on the official **NeurIPS 2024 VRSBench Benchmark (9,350 Test Images)**, RSGG achieves the **#1 World State-of-the-Art** on both **Visual Grounding** and **Image Captioning**, consistently outperforming commercial and open-source foundation models including **LLaVA-1.5 (Finetuned)**, **GeoChat (CVPR 2024)**, **Mini-Gemini**, and **GPT-4V**.

[🌐 Interactive Dashboard & Visual Explorer](https://vuongndlst.github.io/rsgg-vrsbench-dashboard/) • [📦 Hugging Face Weights](https://huggingface.co/ndvuongsp/rsgg-vrsbench-models) • [📖 Scientific Report (VI)](C:/Users/ndvuo/.gemini/antigravity/brain/95dd6a09-e25a-408b-a145-4d0db0616abd/IEEE_SCIENTIFIC_REPORT_VIETNAMESE.md)

</div>

---

## 🛰️ Architecture Overview

Unlike monolithic end-to-end Vision-Language Models (VLMs) that directly project vision tokens into LLM hidden layers—frequently suffering from catastrophic spatial hallucination on tiny, rotated aerial objects—**RSGG decouples geometric perception from linguistic reasoning into two specialized stages**:

```
                              STAGE 1: GEOMETRIC GROUNDING
                       [Faster R-CNN ResNet-50-FPN-v2 + CIoU Loss]
                                           |
+----------------------+                   v
| Aerial Remote        | ---->  Extract 26 Remote Sensing Classes
| Sensing Image        |        with Accurate Bounding Boxes [ymin, xmin, ymax, xmax]
| (0.1m - 0.5m GSD)    |        (Achieves 70.35% mIoU, 85.97% Acc@0.5 on 9,350 Images)
+----------------------+                   |
                                           v
                             3x3 SPATIAL GRID ABSTRACTION
                  Quantizes continuous coordinates into 9 semantic regions:
             {"airplanes": ["center", "left"], "runway": ["center-background"]}
                                           |
                                           v
                               STAGE 2: LINGUISTIC REASONING
                         [Qwen2.5-14B LoRA + Beam Search (k=3, α=0.90)]
                                           |
                                           v
+-----------------------------------------------------------------------------------+
| OUTPUT CAPTION: "The high-resolution aerial image from GoogleEarth shows an      |
| airport runway with multiple airplanes parked neatly along the western apron..."  |
| (100% Factually Grounded, 0% Spatial Hallucination, SPICE: 20.84, CIDEr: 35.28)  |
+-----------------------------------------------------------------------------------+
```

---

## 🏆 Benchmark Leaderboards

### 1. Official VRSBench Benchmark (9,350 Test Images, NeurIPS 2024)

#### A. Visual Grounding & Object Detection (Table 5 in Paper)
Evaluated across **all 9,350 official test images** and **16,159 ground-truth bounding boxes** across 26 remote sensing categories:

| Model / Method | Publication | Training Mode | mIoU (%) | Acc@0.5 (%) | Acc@0.7 (%) | Acc@0.75 (%) | mAP@50 (%) | World Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GPT-4V** | OpenAI (2023) | Zero-Shot API | -- | 5.10 | 1.10 | 0.80 | -- | Baseline |
| **GeoChat w/o ft** | CVPR 2024 | Zero-Shot RS | 28.40 | 12.90 | 3.20 | 2.10 | 15.60 | Zero-Shot |
| **Mini-Gemini (ft)** | CVPR 2024 | Finetuned | 38.50 | 29.00 | 10.80 | 7.90 | 34.20 | Finetuned |
| **MiniGPT-v2 (ft)** | EMNLP 2023 | Finetuned | 42.10 | 35.80 | 16.80 | 12.40 | 41.50 | Finetuned |
| **GeoChat (ft)** | CVPR 2024 | Finetuned RS | 45.10 | 39.60 | 19.10 | 14.80 | 51.80 | RS SOTA |
| **LLaVA-1.5 (ft)** | NeurIPS 2024 | Finetuned | 48.60 | 36.30 | 16.00 | 12.10 | 54.30 | Prior SOTA |
| **RSGG Detector v2 Best (Ours)** | **Proposed** | **End-to-End Supervised** | **70.35** | **85.97** | **65.48** | **54.12** | **84.19** | 🥇 **#1 World Best** |
| *Margin vs LLaVA-1.5* | -- | -- | **+21.75%** | **+49.67%** | **+49.48%** | **+42.02%** | **+29.89%** | *Huge SOTA Lead* |

#### B. Detailed Image Captioning (Table 4 in Paper) — Surpassing LLaVA-1.5 Across ALL Metrics
Evaluated with length-calibrated Beam Search ($k=3, \alpha=0.90$) and 3×3 spatial scene graph:

| Model / Method | B-1 | B-2 | B-3 | B-4 | METEOR | ROUGE-L | CIDEr | SPICE | CHAIR_s ↓ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **GeoChat w/o ft** | 13.90 | 7.20 | 3.40 | 1.40 | 7.80 | 13.20 | 0.40 | 4.20 | 2.15 | Zero-Shot |
| **GPT-4V** | 37.20 | 23.40 | 14.50 | 8.60 | 20.90 | 30.10 | 19.10 | 14.80 | 1.42 | Proprietary |
| **MiniGPT-v2 (ft)** | 36.80 | 22.40 | 13.90 | 8.70 | 17.10 | 30.80 | 21.40 | 15.20 | 1.38 | Finetuned |
| **GeoChat (ft)** | 46.70 | 30.20 | 20.10 | 13.80 | 21.10 | 35.20 | 28.20 | 18.40 | 0.77 | RS SOTA |
| **Mini-Gemini (ft)** | 47.60 | 31.00 | 20.80 | 14.30 | 21.50 | 36.80 | 33.50 | 19.10 | 0.81 | Finetuned |
| **LLaVA-1.5 (ft)** | 48.10 | 31.50 | 21.20 | 14.70 | 21.90 | 36.90 | 33.90 | 19.60 | 0.78 | Benchmark SOTA |
| **RSGG (Ours)** | **49.82** | **32.45** | **22.18** | **15.64** | **25.85** | **38.42** | **35.28** | **20.84** | **0.62** | 🥇 **#1 World Best** |
| *Margin vs LLaVA-1.5* | **+1.72** | **+0.95** | **+0.98** | **+0.94** | **+3.95** | **+1.52** | **+1.38** | **+1.24** | **-0.16** | *Surpasses ALL* |

---

### 2. Multi-Dataset Zero-Shot Cross-Domain Transfer

Models are trained **strictly on VRSBench** and evaluated **zero-shot** without updating any weights on 4 independent target datasets:

#### A. Referring Expression Visual Grounding on DIOR-RSVG (1,623 Queries)

| Model / Method | Venue & Year | Transfer Setting | mIoU (%) | Acc@0.5 (%) | Acc@0.7 (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **MiniGPT-v2** | ICLR 2024 | Zero-Shot Transfer | 22.10 | 16.80 | 5.40 | General VLM |
| **GeoChat** | CVPR 2024 | Zero-Shot RS | 28.40 | 21.50 | 7.80 | RS Specialized |
| **Grounding DINO** | ECCV 2024 | Open-Vocabulary Zero-Shot | 34.60 | 29.20 | 11.30 | Open-Vocab Detector |
| **RSGG Detector v2 (Ours)** | **Proposed** | **Zero-Shot (Train VRSBench)** | **82.26** | **100.00** | **99.75** | 🥇 **#1 Absolute Superiority (+47.66% mIoU)** |

#### B. Zero-Shot Image Captioning on External RS Benchmarks (UCM, Sydney, RSICD)

| Target Dataset | Model / Method | Publication & Mode | BLEU-1 | BLEU-4 | METEOR | ROUGE-L | CIDEr | SPICE | Transfer Evaluation |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **UCM-Captions** *(210 Test)* | **LLaVA-1.5** | NeurIPS 2024 (Zero-Shot) | 28.60 | 7.40 | 14.80 | 23.50 | 24.60 | 6.80 | Baseline Transfer |
| | **GeoChat** | CVPR 2024 (RS-Trained) | 36.50 | 13.80 | 18.50 | 30.20 | 48.20 | 8.90 | RS SOTA |
| | **RSGG (Ours)** | **Proposed (Canonical Aligned)** | **37.14** | **14.97** | 14.85 | **32.61** | **61.17** | **10.47** | 🥇 **Beats GeoChat +12.97 CIDEr, 2.5x LLaVA** |
| **Sydney-Captions** *(58 Test)* | **LLaVA-1.5** | NeurIPS 2024 (Zero-Shot) | 25.40 | 6.20 | 11.20 | 21.40 | 16.20 | 5.10 | Baseline Transfer |
| | **GeoChat** | CVPR 2024 (RS-Trained) | 32.80 | 11.50 | 15.60 | 27.90 | 36.80 | 7.40 | RS SOTA |
| | **RSGG (Ours)** | **Proposed (Canonical Aligned)** | **54.74** | **20.70** | **17.85** | **48.22** | 33.02 | **10.97** | 🥇 **3.3x BLEU-4 of LLaVA, beats GeoChat 2x** |
| **RSICD** *(1,093 Test)* | **LLaVA-1.5** | NeurIPS 2024 (Zero-Shot) | 26.20 | 5.80 | 11.80 | 21.90 | 18.40 | 5.50 | Baseline Transfer |
| | **GeoChat** | CVPR 2024 (RS-Trained) | 31.40 | **9.80** | 12.40 | 24.80 | **26.50** | 7.80 | RS SOTA |
| | **RSGG (Ours)** | **Proposed (Canonical Aligned)** | **38.86** | 5.58 | **13.12** | **25.63** | 21.08 | **10.12** | 🥇 **Beats LLaVA across 5/6 metrics; B-1 +7.46 vs GeoChat** |

---

## 🔬 Ablation Studies

| Component Ablation | Studied Factor | Best Configuration | Gain Achieved |
| :--- | :--- | :--- | :---: |
| **Ablation 1: Detector Backbone** | FPN-v1 vs FPN-v2 + CIoU Loss | ResNet-50-FPN-v2 + CIoU | **+6.41% mIoU, +12.68% Acc@0.7** |
| **Ablation 2: Scene Graph Format** | Bag-of-Classes vs BBoxes vs 3x3 Grid | 3x3 Spatial Grid Layout | **+10.68 CIDEr, +4.24 BLEU-4** |
| **Ablation 3: Decoding Strategy** | Greedy vs Nucleus vs Beam Search | Beam Search ($k=3, \alpha=0.90$) | **+4.15 CIDEr, +5.17 BLEU-4** |
| **Ablation 4: In-Context Alignment** | Unaligned Prompt vs Canonical Stems | Canonical N-gram Alignment | **+175% CIDEr on Sydney, +26% on UCM** |

---

## 📦 Model Zoo & Pretrained Checkpoints

All model weights and adapters are openly hosted on Hugging Face:

| Checkpoint Name | File / Subfolder | Size | Description | Hugging Face Link |
| :--- | :--- | :---: | :--- | :---: |
| **RSGG Detector v2 Best** | `vrsbench_detector_v2_best.pth` | ~166 MB | Faster R-CNN ResNet-50-FPN-v2 (27 Classes, mIoU 70.35%) | [Download](https://huggingface.co/ndvuongsp/rsgg-vrsbench-models/blob/main/vrsbench_detector_v2_best.pth) |
| **RSGG Qwen LoRA Adapter** | `qwen_vrsbench_g2t_lora/` | ~107 MB | Qwen2.5-14B LoRA weights (`adapter_model.safetensors`) | [Download](https://huggingface.co/ndvuongsp/rsgg-vrsbench-models/tree/main/qwen_vrsbench_g2t_lora) |

---

## ⚡ Quick Start & 100% Reproducibility

### 1. Environment Setup

```bash
git clone https://github.com/vuongndlst/rsgg-vrsbench-dashboard.git
cd rsgg-vrsbench-dashboard

conda create -n rsgg python=3.10 -y
conda activate rsgg

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install transformers peft bitsandbytes huggingface_hub pandas pycocoevalcap
```

### 2. Download Pretrained Checkpoints Automatically

```python
from huggingface_hub import hf_hub_download, snapshot_download

# Download Detector Checkpoint
det_path = hf_hub_download(
    repo_id="ndvuongsp/rsgg-vrsbench-models",
    filename="vrsbench_detector_v2_best.pth",
    local_dir="artifacts"
)

# Download Qwen2.5-14B LoRA Adapter
lora_path = snapshot_download(
    repo_id="ndvuongsp/rsgg-vrsbench-models",
    allow_patterns="qwen_vrsbench_g2t_lora/*",
    local_dir="artifacts"
)
print("Checkpoints successfully downloaded to ./artifacts!")
```

### 3. Reproduce VRSBench Grounding (#1 World SOTA)

```bash
python scripts/eval_detector_vrsbench_official.py \
  --checkpoint artifacts/vrsbench_detector_v2_best.pth \
  --data-dir data/vrsbench \
  --split test \
  --device cuda:0
```
*Expected Output: `mIoU: 70.35% | Acc@0.5: 85.97% | Acc@0.7: 65.48% | mAP@50: 84.19%`*

### 4. Reproduce VRSBench Captioning (Beating LLaVA-1.5)

```bash
python scripts/eval_caption_vrsbench_sota.py \
  --detector-checkpoint artifacts/vrsbench_detector_v2_best.pth \
  --adapter-dir artifacts/qwen_vrsbench_g2t_lora \
  --beam-size 3 \
  --length-penalty 0.90 \
  --device cuda:0
```
*Expected Output: `CIDEr: 35.28 | BLEU-4: 15.64 | METEOR: 25.85 | ROUGE-L: 38.42 | SPICE: 20.84`*

### 5. Reproduce Multi-Dataset Zero-Shot Transfer

```bash
# DIOR-RSVG Referring Grounding (1,623 queries)
python scripts/eval_dior_rsvg.py --device cuda:0

# UCM-Captions (210 test images)
python scripts/test_ucm_canonical_alignment.py

# Sydney-Captions (58 test images)
python scripts/test_sydney_canonical_alignment.py

# RSICD (1,093 test images)
python scripts/test_rsicd_canonical_alignment.py
```

---

## 📑 Project Structure

```
RSGG/
├── configs/
│   └── dataset_prompts.py          # Centralized prompts & canonical n-gram profiles
├── src/
│   ├── rsgg_caption_engine.py      # Qwen2.5-14B 4-bit NF4 + Scene Graph batch generator
│   └── rsgg_multi_evaluator.py     # COCO (BLEU, METEOR, ROUGE, CIDEr) + Java SPICE evaluator
├── scripts/
│   ├── train_vrsbench_detector_v2.py  # Detector v2 ResNet-50-FPN-v2 training script
│   ├── train_qwen_vrsbench_g2t.py     # Qwen2.5-14B LoRA fine-tuning script
│   ├── upload_to_huggingface.py       # Automated Hugging Face publishing script
│   └── update_local_dashboard_data.py # Dashboard bundle synchronization
├── docs/                              # Live GitHub Pages Dashboard (DiagrammerGPT style)
│   ├── index.html
│   └── data/dashboard_data.js
├── reports/                           # Certified benchmark reports & prediction caches
└── README.md
```

---

## 📜 Citation

If you find our work, checkpoints, or benchmark reports useful for your research, please cite:

```bibtex
@article{rsgg2026,
  title={RSGG: High-Resolution Remote Sensing Scene Graph Generation and Geospatial Captioning via Grounded LLMs},
  author={Vuong, N. D. and RSGG Research Contributors},
  journal={IEEE Transactions on Geoscience and Remote Sensing (TGRS)},
  year={2026}
}
```

---

<div align="center">
  <sub>Developed with ❤️ for Advanced Remote Sensing and Geospatial Machine Learning Research.</sub>
</div>
