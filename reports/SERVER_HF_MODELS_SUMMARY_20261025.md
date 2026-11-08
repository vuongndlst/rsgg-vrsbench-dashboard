# Server Hugging Face model inventory — 25/10/2026

Quét chỉ đọc sau khi xóa FAIR. Có weights không đồng nghĩa đã kiểm tra load/inference. Size là tổng weight files tại một vị trí, không phải VRAM hoặc tổng unique disk cả model family.

| Model | Weights GiB / vị trí điển hình | Công dụng |
|---|---:|---|
| Qwen/Qwen2.5-14B-Instruct | 27.511 | text LLM;8/8shards |
| Qwen2.5-VL-7B-Instruct | 15.445 | vision-language;5/5shards |
| openai/clip-vit-base-patch32 | 0.564 | image-text encoder |
| GroundingDINO-Tiny | 0.642 | open-vocabulary detector; local directory name/model config |
| google/owlv2-base-patch16-ensemble | 0.577 | open-vocabulary detector |
| nvidia/segformer-b2-finetuned-ade-512-512 | 0.102 | semantic segmentation;two revisions/formats observed |
| MCG-NJU/videomae-base-finetuned-kinetics | 0.322 | video encoder/classification;other HF snapshot also present |
| microsoft/git-base-vatex | 0.658 | video captioning |
| GIT-large | 1.469 | captioning;split local folder variants below |
| GIT-large-COCO | 1.469 | image captioning |
| vinai/phobert-base-v2 | 0.503 | Vietnamese language encoder;project safetensors copy also present |
| timm/vit_small_patch14_dinov2.lvd142m | 0.082 | visual encoder;weights found, not full runtime validation |

## Đường dẫn model trên server

**Qwen/Qwen2.5-14B-Instruct**
- `${REMOTE_HOME}/shared-assets/models/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8` — 27.511GiB/8weight files

**Qwen2.5-VL-7B-Instruct**
- `${REMOTE_HOME}/etechs/ouc_stkg_captioning_research_pack/artifacts/multimodal-labels-v1/models/Qwen2.5-VL-7B-Instruct` — 15.445GiB/5weight files

**openai/clip-vit-base-patch32**
- `${REMOTE_HOME}/shared-assets/models/clip-vit-base-patch32` — 0.564GiB/1weight files

**GroundingDINO-Tiny**
- `${REMOTE_HOME}/etechs/msvd-tkg-captioning/models/grounding-dino-tiny` — 0.642GiB/1weight files

**google/owlv2-base-patch16-ensemble**
- `${REMOTE_HOME}/.cache/huggingface/hub/models--google--owlv2-base-patch16-ensemble/snapshots/cfd3195ba4ea9592eec887ded089f4c08eff231d` — 0.577GiB/1weight files

**nvidia/segformer-b2-finetuned-ade-512-512**
- `${REMOTE_HOME}/.cache/huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512/snapshots/de01bae28967510f9ddd496c60a969357195400c` — 0.102GiB/1weight files
- `${REMOTE_HOME}/.cache/huggingface/hub/models--nvidia--segformer-b2-finetuned-ade-512-512/snapshots/4585665b1bf59b90b831d5145d3e25d6e0743d03` — 0.102GiB/1weight files

**MCG-NJU/videomae-base-finetuned-kinetics**
- `${REMOTE_HOME}/etechs/msvd-tkg-captioning/models/videomae-kinetics` — 0.322GiB/1weight files

**microsoft/git-base-vatex**
- `${REMOTE_HOME}/etechs/ouc_stkg_captioning_research_pack/artifacts/sav-tkg/sav140-supervised-p1-isolated-v2/git-baseline-20261022-v1/model_cache/models--microsoft--git-base-vatex/snapshots/623363cef8744c57438ce8d62b30bef1fbbec111` — 0.658GiB/1weight files

**GIT-large**
- `${REMOTE_HOME}/models/git-large` — 1.469GiB/1weight files

**GIT-large-COCO**
- `${REMOTE_HOME}/models/git-large-coco` — 1.469GiB/1weight files

**vinai/phobert-base-v2**
- `${REMOTE_HOME}/.cache/huggingface/hub/models--vinai--phobert-base-v2/snapshots/86cd7fd4c148980922ac11a2cf5e257f2ba639e1` — 0.503GiB/1weight files

**timm/vit_small_patch14_dinov2.lvd142m**
- `${REMOTE_HOME}/etechs/thesis/var/huggingface/hub/models--timm--vit_small_patch14_dinov2.lvd142m/snapshots/4610ca143709d58a633b6397a74412c2c3842454` — 0.082GiB/1weight files

## Bản Qwen14B riêng thứ hai

`${REMOTE_HOME}/etechs/msvd-tkg-captioning/models/qwen2.5-14b` có8/8shards, khoảng27.51GiB, inode khác shared-assets. Chưa SHA-verify nội dung weights; chưa hợp nhất/xóa. Cần kiểm tra dependencies và process trước khi cân nhắc.

## Metadata và giới hạn

Qwen2.5-7B hiện chỉ thấy config/index,0/4shards tại các thư mục đã quan sát. Các BERT configs trong SwinBERT source không được đếm thành model đã tải đủ. Không tính test fixtures hay training checkpoints vào danh sách HF ở trên.

Scan hoàn tất25.52s trên filesystem truy cập được; có permission-denied ở các vùng hệ thống/root. Không tải thêm model, không đọc HF tokens, không load weights. Directory symlinks không được traverse toàn bộ; không bảo đảm kiểm kê các vùng không có quyền.

FAIR đã xóa: giải phóng6.78GiB; free ngay sau xóa14.77GiB. Reserve16GiB còn thiếu1.23GiB ở snapshot này.