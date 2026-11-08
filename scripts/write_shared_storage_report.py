"""Render the concrete storage result and reviewable next cleanup candidates."""
from pathlib import Path
import json
root=Path(__file__).absolute().parents[1]
read=lambda n:json.loads((root/'reports'/n).read_text(encoding='utf-8'))
model=read('SERVER_MODEL_CLEANUP_RESULT.json');assets=read('SERVER_KARPATHY_MOVE_RESULT.json');verify=read('SERVER_SHARED_ASSETS_VERIFICATION.json');audit=read('SERVER_ASSET_DETAIL_AUDIT.json');pre=read('SERVER_CLEANUP_PREFLIGHT.json')
gib=lambda n:f'{n/2**30:.2f}'
home_rows={r['path']:r['allocated_bytes'] for r in verify['home_sizes']['rows']};fair_rows={r['path']:r['allocated_bytes'] for r in verify['fair_sizes']['rows']}
removed=[e for e in model['events'] if e['action']=='removed']
size_map={x['path']:x.get('allocated_bytes',0) for x in pre['delete_targets']}
size_map['${REMOTE_HOME}/etechs/final_project/var/models/clip-vit-base-patch32']=pre['moves'][0]['allocated_bytes']
rows='\n'.join(f"| `{x['path']}` | {gib(size_map[x['path']])} | {'Đã xóa weights trùng; giữ alias tới bản chung' if x['compatibility_alias'] else 'Đã xóa'} |" for x in removed)
project_names=['fair-project','coco_kb_v3','v2_kb_20260909','v2_kb','skgcap_vlm','image-captioning-kg','etechs','rsg-former','sbir','models','spice_setup']
project_rows='\n'.join(f"| `~/{n}` | {gib(home_rows['${REMOTE_HOME}/'+n])} | {'Giữ VG đang chạy + cache/checkpoints' if n=='rsg-former' else 'Giữ VG ZIP' if n=='sbir' else 'Xem phân tích dưới; chưa xóa project'} |" for n in project_names if '${REMOTE_HOME}/'+n in home_rows)
archive=[x for x in audit['backup_candidates'] if any(t in x['path'].lower().split('/')[-1] for t in ['backup','archive','src_old'])]
archive_rows='\n'.join(f"| `{x['path']}` | {gib(x['sizes']['rows'][0]['bytes'])} |" for x in archive)
fair_candidates=['output_sg2text/cache','output_captions_v2/cache','third_party/T2I-CompBench-main','third_party/semantic-object-accuracy','output_images_v2','scripts/phase2_text2img_ver2']
fair_candidate_rows='\n'.join(f"| `~/fair-project/{n}` | {gib(fair_rows['${REMOTE_HOME}/fair-project/'+n])} |" for n in fair_candidates if '${REMOTE_HOME}/fair-project/'+n in fair_rows)
dataset_rows='\n'.join(f"| {n} | {x['available_test_images']}/{x['expected_test_images']} | {x['human_reference_captions']} | {verify['datasets'][n]['images_header_verified']} | {'Đủ ảnh test' if verify['datasets'][n]['full_test_ready'] else 'Thiếu '+str(x['missing_test_images'])+' ảnh; không chấm full test'} |" for n,x in assets['datasets'].items())
report=f'''# Kết quả dọn server và dữ liệu dùng chung — 10/10/2026

Người dùng đã trực tiếp cho phép xóa các model CLIP trùng, LLaVA-OneVision-Qwen2-7B, SD3 medium, SD3 ControlNet Canny, SigLIP-so400m, Qwen2.5-VL-3B; sau đó xác nhận dừng `batch_generate.py` của fair. Chỉ các model trong danh sách đó bị xóa. Các dự án/backup khác dưới đây là **đề xuất**, chưa bị xóa.

## Đã thực hiện và kiểm tra

- Dừng đúng PID 2916729, script `batch_generate.py`, owner và cwd `fair-project` đã xác minh; dùng pidfd và SIGTERM. Không cần SIGKILL. Process có tuổi khoảng 47,17 ngày, giữ 3.535.572.992 allocated bytes (~3,29 GiB) của `.incomplete (deleted)`; dung lượng trả lại sau khi process đóng FD. Không dừng Django `manage.py` PID 648773 hoặc VG.
- Model/cache dọn có free-space gain đo tại thao tác **{gib(model['observed_free_gain_bytes'])} GiB**. Cộng file do process cũ giữ và metadata COCO trùng đã consolidate: khoảng **{gib(verify['free_bytes']-pre['free_bytes'])} GiB** tăng so với preflight. Chênh lệch df có thể gồm thay đổi nhỏ từ các job dùng chung.
- Persistent filesystem hiện còn **{gib(verify['free_bytes'])} GiB** theo lần kiểm tra `{verify['at_utc']}`. Chuyển cùng filesystem bằng rename, không tạo một bản Qwen 27,5 GiB thứ hai.
- Qwen 14B đã chuyển, đủ 8/8 shard BF16. Inode/size của từng shard giữ nguyên; index SHA và safetensors layout hợp lệ. Chưa full-weight hash/GPU inference; không mở GPU job mới.
- Chỉ còn một bản CLIP ViT-B/32 vật lý, đầy đủ image processor, tokenizer, vocab, merges. SHA256 weights `{model['clip_weight_sha256']}`; các bản flat giữ alias tương thích. Shared HF repository chỉ là metadata/snapshot link, không có weights thứ hai. CLIP ViT-L/14-336 và OpenAI `.pt` đã xóa; API cũ yêu cầu đúng variant/format đó cần chuyển loader sang bản HF chung, không giả chúng tương đương.
- VG pipeline active: **{verify['VG_pipeline_active']}**, PID {verify['VG_pipeline_pid']}; config SHA và toàn bộ {len(verify['VG_locked_sources_match'])} source files khớp preregistration; VG ZIP còn nguyên. Không chạy evaluation, test hoặc thay training config.
- Initial automatic review đã chặn chuyển Qwen khi old fair process còn dùng cache. Sau xác nhận trực tiếp của người dùng, process đã dừng; chuyển Qwen kiểm tra không còn consumer trước mutation và được duyệt. Không còn blocker này.

[Cleanup result](../reports/SERVER_MODEL_CLEANUP_RESULT.json), [stop process](../reports/SERVER_FAIR_BATCH_STOP_RESULT.json), [post-verification](../reports/SERVER_SHARED_ASSETS_VERIFICATION.json).

## Model đã dọn

Số trong bảng là allocated bytes trước dọn của directory/file, không phải tổng sẽ cộng khi có inode/link dùng chung. Result JSON giữ exact paths và actions.

| Path alias | GiB trước dọn | Kết quả |
|---|---:|---|
{rows}

Giữ Qwen2.5-VL-7B, BLIP, GIT, GIT VG conditioned/plain, OWLv2, RelTR và các trained project checkpoints ngoài danh sách đã duyệt. Giữ dependencies, Java/SPICE, `.local`, `.pyenv`, VG cache và datasets khác.

## Chỗ dùng chung

```text
~/shared-assets/
  ASSET_REGISTRY.json
  models/
    clip-vit-base-patch32/
    huggingface/hub/
      models--Qwen--Qwen2.5-14B-Instruct/
      models--openai--clip-vit-base-patch32/  # metadata + snapshot alias
  datasets/
    coco_karpathy/
      annotations/dataset_coco.json
      images/
      test_manifest.json
      test_references_coco.json
      missing_test_images.json
    flickr30k_karpathy/
      annotations/dataset_flickr30k.json
      images/
      test_manifest.json
      test_references_coco.json
      missing_test_images.json
  maintenance/
```

Canonical Qwen snapshot: `{model['qwen_shared_snapshot']}`. Dùng đường dẫn này với `local_files_only=True` sau runtime pilot; không trỏ vào default HF metadata stub hoặc download lại. Đường dẫn Qwen cũ trong fair, CLIP flat cũ, metadata COCO/Flickr và image directories nguồn giữ symlink tương thích. Không thay global HF_HOME hoặc shell environment; registry dùng `${{REMOTE_HOME}}` placeholder, resolve bằng `Path.home()` trên server. Registry được lưu thêm tại `~/rsg-former/reports/SHARED_ASSET_REGISTRY.json` và [workspace copy](../reports/SHARED_ASSET_REGISTRY.json).

## Karpathy test và khoảng trống dữ liệu

| Dataset | Ảnh test hiện có/đúng split | Human reference captions | Image.verify đạt | Trạng thái |
|---|---:|---:|---:|---|
{dataset_rows}

COCO metadata SHA `2fd999220673258012acfb411a4e7e66af7d488050b2519b0badcc49b7600b8d`; Flickr30k metadata SHA `db779ee2c5df40c25ff0eebeca464f6124b620cc4a9398650aea71fd1b53d4af`. COCO toàn metadata train82.783/restval30.504/val5.000/test5.000; ảnh di chuyển 5.503 (gồm ảnh ngoài test), **không có đủ ảnh full train/val**. Flickr metadata train29.000/val1.014/test1.000; directory di chuyển có 1.000 JPEG nhưng chỉ 34 thuộc đúng Karpathy test. Không suy ra provenance hoặc test membership từ tên thư mục.

Audit filename đã chạy hết trên filesystem có quyền đọc, xét 822.177 JPEG, không follow directory symlinks; 22 permission errors giới hạn các khu vực không đọc được. Đây không phải tuyên bố đầy đủ mọi tài khoản/archives. [Image discovery](../reports/SERVER_KARPATHY_IMAGES_AUDIT.json). Image.verify kiểm tra cấu trúc/header, chưa decode toàn bộ pixel hoặc khóa hash ảnh để preregistration evaluation.

Đã giữ danh sách exact 966 filename thiếu trong `~/shared-assets/datasets/flickr30k_karpathy/missing_test_images.json`; cần bổ sung từ Flickr30k đúng nguồn trước full caption evaluation. Nguồn ảnh chính thức: [Flickr30k dataset](https://shannon.cs.illinois.edu/DenotationGraph/data/index.html), [official image archive link](https://uofi.box.com/s/1cpolrtkckn4hxr1zhmfg0ln9veo6jpl). Link Box không đọc được qua web tool ở phiên này; chưa tải archive hoặc dùng mirror thay thế. Không thay bằng Flickr8k, random1k hoặc tự sửa test split.

`test_references_coco.json` là chuyển human captions sang COCO API shape, không phải model outputs; COCO có25.010 refs (không giả định đúng5 mọi ảnh), Flickr có5.000. References chỉ dùng chấm pycocoevalcap, không đưa vào graph/Qwen input. Captioning chưa chạy; overlap với VG train, model inference pilot, prompt lock, full image provenance vẫn là N1/N2 gates.

## Fair: giữ phần chính nào

Sau chuyển Qwen và dọn model đã duyệt, `~/fair-project` còn **{gib(home_rows['${REMOTE_HOME}/fair-project'])} GiB** theo du không follow model alias. Đề xuất giữ:

1. Code `scripts/phase3_img2text_v2`, `scripts/phase4_sg2text`, `scripts/evaluation`, configs, environment/version manifests, evaluation protocols và scripts xuất caption. Với phase2 text-to-image đã kết thúc, giữ code/config cần tái lập trong paper archive trước khi xét bỏ vendor/cache.
2. Các kết quả cuối, per-image captions/metrics, ablation results, split/train manifests, seeds, checkpoint hashes, logs cần provenance. Không chỉ giữ một bảng metric để rồi mất traceability.
3. `output_sg2text/trained`: hai GIT VG conditioned/plain (~2,94 GiB) là trained research artifacts, không tự coi là duplicates. Có thể dùng baseline riêng sau overlap audit; giữ cho tới khi quyết định không cần hoặc chuyển archive lưu dài hạn.
4. Metadata datasets/dictionaries, configs mappings và `datasets/vg_para`, `vs_genome`, `kg_v2`, `kg_v3` cần đối chiếu nghiên cứu. Qwen đã độc lập fair; dữ liệu Karpathy dùng canonical shared root.
5. Paper FAIR cuối và revision evidence ở `~/etechs/final_project/papers/final_fair/`; `final_project` còn Django `manage.py` đang chạy, không tự gộp nó vào project fair đã kết thúc.

Các paths có thể dọn tiếp **sau khi xác nhận chúng không chứa bằng chứng cuối hoặc consumer**:

| Fair candidate | GiB hiện đo |
|---|---:|
{fair_candidate_rows}

Cache có thể tái tạo ưu tiên hơn trained checkpoints. `scripts/phase2_text2img_ver2` có thể chứa assets bên trong dù tên là scripts; cần file manifest trước khi dọn, không xóa cả source vì folder lớn. Generated images có thể là figure/evidence đã dùng trong paper: archive bản cuối trước dọn. Toàn bộ candidates bảng trên **chưa xóa**.

## Projects ở home root và các mục ngoài home

| Root project | GiB hiện đo | Chính sách hiện tại |
|---|---:|---|
{project_rows}

`~/etechs` là nhóm nhiều project, không phải một backup. Snapshot chi tiết trước dọn: `tien`62,46GiB (student_behavior55,77GiB, venv6,46GiB); `ouc_stkg_captioning_research_pack`72,19GiB trước bỏVL3 ~6,99GiB; `thesis`22,45GiB với artifacts21,73GiB. Đây là nơi ưu tiên kiểm kê **cache/features/videos/checkpoint retention theo từng project**, chưa đủ căn cứ xóa project. `final_project` có service sống và paper revision; không xóa nguyên cây.

`v2_kb_20260909/outputs` khoảng3,60GiB, sau bỏCLIP cache có thể cân nhắc archive kết quả cuối rồi bỏ các lần trung gian. `coco_kb_v3` khoảng5–6GiB trước dọn phần CLIP/metadata: giữ BLIP baseline, manifests/exclusion audit và final outputs; xem các repeated candidate/ablation outputs theo từng run. `v2_kb`0,57GiB: xem old/vendor/derived KB, không đủ lớn để ưu tiên trước tens-of-GiB feature caches. `skgcap_vlm` đã chuyển COCO assets ra shared, còn graph/artifacts riêng; không xóa như duplicate fair chỉ vì có kgproj copied code.

`/tmp` khoảng3,97GiB ở audit: `charades_lighthouse_features`1,80GiB, `ouc-label-browsers`0,64GiB, `conquer_official`0,33GiB và nhiều tools nhỏ. Chưa xóa vì có thể phục vụ project khác. `/openssh-9.8p1` và tar ở `/` là source/admin artifact, không phải project captioning của tài khoản; không tự dọn. `/usr`, `/opt`, `/var` và các tài khoản khác không được coi là user project trash.

Giữ `spice_setup` (CoreNLP/JRE) cho pycocoevalcap/SPICE. Chỉ hai compressed installers `corenlp.zip`403.157.240bytes và `jre8.tgz`41.383.497bytes là candidate thu hồi ~0,41GiB sau xác minh bản giải nén đầy đủ; runtime Java/CoreNLP không dọn. Dependencies `.local/.pyenv` đang có consumers và training overlay: không xóa mù.

## Backup cũ

Chỉ liệt kê names thực có backup/archive/src_old; các hits như `gold`, `threshold`, `phone_holding` của glob `*old*` không phải chứng cứ backup.

| Backup/archive candidate | GiB hiện đo |
|---|---:|
{archive_rows}

Backup paper FAIR lớn nhất nhóm này khoảng80MiB; còn đa số archive rất nhỏ. Lợi ích disk thấp hơn nhiều so với model cleanup đã thực hiện. Đề xuất giữ bản paper/revision cuối và ít nhất một bản nguồn phục hồi; chưa xóa backup nào. Metadata `disk-cleanup-20260925` chỉ~0,37MiB và là audit ledger nên giữ.

## Bước tiếp theo

- Bổ sung đúng966ảnh Flickr30k Karpathy test; khóa image manifest/hash/provenance. COCO không cần tải lại full archive để chấm5.000test.
- Captioning Qwen runtime/prompt và evaluator vẫn chờ VG gate, không dùng dataset tổ chức xong làm lý do mở test/tune sớm.
- Dọn vòng2 theo danh sách project-specific caches/outputs sau xem KEEP manifest; không xóa nguyên `etechs` hoặc fair để đạt sốGiB tùy ý.
- STAR archive112,44GiB vẫn lớn hơn free disk hiện có cộng reserve16GiB; cần ít nhất~45GiB bổ sung cho archive+reserve, chưa tính giải nén/artifacts. Không giải nén VG.
'''
(root/'research/SERVER_STORAGE_CLEANUP.md').write_text(report,encoding='utf-8')
plan=root/'research/NEXT_PHASE_PLAN.md'
text=plan.read_text(encoding='utf-8').replace('${REMOTE_HOME}/fair-project/models/hf_cache/hub/models--Qwen--Qwen2.5-14B-Instruct/snapshots/cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8',model['qwen_shared_snapshot'])
notice='\nCập nhật storage 10/10: [cleanup và shared-assets](SERVER_STORAGE_CLEANUP.md), [registry](../reports/SHARED_ASSET_REGISTRY.json). Qwen đã chuyển sang shared root, đủ8shards; CLIP giữ một bản. COCO Karpathy đủ5.000test images; Flickr30k chỉ34/1.000, phải bổ sung966 trước full test. Model/old fair process đã dọn theo yêu cầu; chưa mở GPU/caption evaluation.\n'
lines=text.splitlines(keepends=True);lines.insert(1,notice);plan.write_text(''.join(lines),encoding='utf-8')
recommend=root/'research/MODEL_STORAGE_RECOMMENDATIONS.md';old=recommend.read_text(encoding='utf-8');lines=old.splitlines(keepends=True);lines.insert(1,'\n**Cập nhật sau yêu cầu xóa:** model đã dọn và Qwen/CLIP đã chuyển; xem [kết quả hiện tại](SERVER_STORAGE_CLEANUP.md). Các bảng dưới là snapshot trước cleanup, không còn là action list/current existence.\n');recommend.write_text(''.join(lines),encoding='utf-8')
for name in ['README.md','research/README.md']:
    p=root/name;content=p.read_text(encoding='utf-8');content+='\nStorage dùng chung: [cleanup/model/datasets]('+('research/' if name=='README.md' else '')+'SERVER_STORAGE_CLEANUP.md). COCO Karpathy5.000test đủ; Flickr30k34/1.000, chưa được chấm full test. Registry tại reports/SHARED_ASSET_REGISTRY.json.\n';p.write_text(content,encoding='utf-8')
print(json.dumps({'report':'research/SERVER_STORAGE_CLEANUP.md','free_GiB':gib(verify['free_bytes']),'VG_active':verify['VG_pipeline_active'],'datasets':verify['datasets']},ensure_ascii=False))
