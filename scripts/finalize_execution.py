"""Create the current P0-P2 evidence lock, keeping the initial P0 audit immutable."""
from pathlib import Path
import hashlib,json,csv
from datetime import datetime,timezone

ROOT=Path(__file__).absolute().parents[1]
def read(name):return json.loads((ROOT/name).read_text(encoding='utf-8'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    audit=read('reports/P0_RELEASE_AUDIT.json');prep=read('reports/P0_PREPARATION.json')
    env=read('reports/P0_ENVIRONMENT_LOCK.json');storage=read('reports/P0_STORAGE_AUDIT.json')
    tests=read('reports/P2_TEST_REPORT.json');pilot=read('reports/P2_PILOT_REPORT.json')
    config=read('configs/p2_pilot.json');ids=read('data/vg150/split_ids.json')
    assert tests['status']=='PASS' and tests['official_metric_parity_cases']==32
    assert pilot['status']=='pilot_complete' and not pilot['test_split_scored']
    assert pilot['config_sha256']==sha(ROOT/'configs/p2_pilot.json')
    for split,values in ids.items():
        assert len(values)==audit['splits'][split]['images']
        digest=hashlib.sha256(json.dumps(values,separators=(',',':')).encode()).hexdigest()
        assert digest==audit['splits'][split]['ordered_image_ids_sha256']
        if split in pilot['split_hashes']:assert digest==pilot['split_hashes'][split]
    result={
      'updated_utc':datetime.now(timezone.utc).isoformat(),
      'canonical_current_status':True,
      'historical_records':['research/P0_LOCK.json','research/P0_AUDIT_REPORT.md'],
      'authorization':'research/EXECUTION_AUTHORIZATION.json',
      'phase_gates':{'P0_natural':'PASS_for_bounded_shared_execution','P0_satellite':'annotation_release_locked_images_deferred_storage','P1':'PASS_method_controls_and_preregistration','P2_natural':'PASS_engineering_baseline_validation_smoke','P2_satellite':'not_run','P3_plus':'not_started'},
      'dataset':{'image_root':'${REMOTE_HOME}/sbir/hienlth/vg','metadata_root':'${REMOTE_HOME}/rsg-former/data/vg150','image_archives_extracted':False,'releases':prep['files'],'splits':audit['splits'],'archive_coverage':audit['archive_coverage'],'hdf5_provenance':'pinned author HF mirror; reference loader train-index parity verified; original toolkit HDF5 byte identity not established','ignored_split':'split_GLIPunseen','raw_region_json_at_image_root':'truncated; not used'},
      'environment':env,'evaluator':tests,'resource_policy':audit['resources'],
      'STAR_storage':storage,
      'pilot':{'config':config,'config_sha256':pilot['config_sha256'],'device':pilot['device'],'wall_seconds':pilot['wall_seconds'],'gpu_hours_wall_upper_bound':pilot['gpu_hours_upper_bound'],'peak_torch_allocated_bytes':pilot['peak_gpu_allocated_bytes'],'peak_torch_reserved_bytes':pilot['peak_gpu_reserved_bytes'],'artifacts_bytes':pilot['artifact_bytes'],'weights':pilot['weights'],'runs':{k:{'gc':v['metrics']['gc'],'selected_epoch':v.get('selected_epoch'),'checkpoint':v.get('checkpoint'),'checkpoint_sha256':v.get('checkpoint_sha256')} for k,v in pilot['runs'].items()},'test_scored':False},
      'limits':['pilot uses deterministic 64 train / 32 validation images; no converged full baseline or SOTA claim','frequency prior sees all train annotations whereas A0/A1 learn from 64 images','STAR full images and satellite adapter/training not available yet','shared GPU; external reservation/time/storage entitlement not observable','PyTorch allocator measurements exclude CUDA context, cuDNN and other jobs','archive ID coverage audited; full archive CRC/decode of every JPEG not performed','environment inherits shared packages; not a fully self-contained frozen environment']}
    rows=[]
    for k,v in pilot['runs'].items():
        m=v['metrics']['gc'];rows.append(f"| {k} | {m['R@50']*100:.2f} | {m['mR@50']*100:.2f} | {v.get('selected_epoch','—')} |")
    report=f'''# P0–P2: kết quả thực thi và các gate còn lại

Ngày 10/10/2026. **P0 natural, P1 và P2 engineering validation smoke đã đạt gate.** P0 satellite đã khóa annotation nhưng chưa đủ storage cho ảnh STAR. Đây là báo cáo trạng thái hiện tại; `P0_LOCK.json` và `P0_AUDIT_REPORT.md` là snapshot audit read-only trước khi người dùng trực tiếp cho phép download/implementation/pilot.

## Dữ liệu P0 đã bổ sung và khóa

VG images giữ nguyên tại `${{REMOTE_HOME}}/sbir/hienlth/vg`: `images.zip`, `images2.zip`. Adapter đọc JPEG trực tiếp qua ZIP, không giải nén ảnh ra filesystem. Đã tải các file thiếu vào `${{REMOTE_HOME}}/rsg-former/data/vg150`: HDF5 VG150, dict 150 object/50 predicate và ordered `image_data.json` của toolkit. Download URLs, bytes và SHA256 tại [P0_PREPARATION.json](../reports/P0_PREPARATION.json).

HDF5 là mirror public của tác giả LLM4SGG, pinned HF revision `73ff29cb7587ab5cf011772e863172a6fa468000`, SHA256 `ad52a01f8e6142bb7bc62d91f97dad1567f44772a23eac68d60a5b7766ffe4b1`. Dict/order/split/evaluator tham chiếu KaihuaTang toolkit commit `{tests['reference_commit']}`. Đã xác minh schema, vocab, train-index parity và content samples; chưa chứng minh HDF5 byte-identical với original distribution của toolkit. Dùng standard `split`; bỏ qua `split_GLIPunseen` trong mirror.

| Split | Ảnh | Objects | Relations |
|---|---:|---:|---:|
| Train | 56.224 | 658.971 | 371.716 |
| Validation | 5.000 | 62.754 | 33.203 |
| Test | 26.446 | 325.570 | 183.642 |

Không overlap ID giữa ba split; không thiếu archive image ID trong 87.670 ảnh cần cho các split. Đã loại bốn corrupt image IDs theo reference. Train dùng overlap filtering ở native 1024 scale; val/test giữ multi-predicate GT. Chuyển box giữ phép integer truncation của loader tham chiếu, có regression test cho odd-width boxes. Sau clipping train có hai graph mất toàn bộ relation; giữ nguyên indexing đúng reference và báo rõ trong [release audit](../reports/P0_RELEASE_AUDIT.json).

Các ordered ID hashes được khóa trong [status JSON](P0_P1_P2_STATUS.json); danh sách đầy đủ lưu local/server tại `data/vg150/split_ids.json`. Coverage metadata không thay thế full ZIP CRC hoặc decode mọi JPEG. Pilot đã decode và kiểm tra dimension cho 96 ảnh sử dụng. Raw `region_descriptions.json` ở VG root bị cắt dở, không dùng; không trộn raw object ID với benchmark HDF5 row ID.

## Environment và resource policy

Đã tạo project SSH `${{REMOTE_HOME}}/rsg-former`. Interpreter `${{REMOTE_HOME}}/rsg-former/.venv/bin/python`: Python3.10.12, torch2.12.0+cu130, torchvision0.27.0+cu130, numpy2.2.6, h5py3.15.1. h5py ở project `.tools/deps`; không đổi package của project khác. Đây là overlay kế thừa torch/system packages được khóa version, chưa phải environment tự chứa hoàn toàn. [Environment lock](../reports/P0_ENVIRONMENT_LOCK.json).

A40 dùng chung; không suy ra allocation độc quyền từ VRAM trống. Người dùng trực tiếp cho phép bounded P0–P2 execution. Chính sách: ≤20GPU-h cho P2, pilot≤1800s, PyTorch CUDA allocator≤8GiB, CPUthreads4, generated artifacts≤15GiB, test không tuning. Đây là execution cap của project; quota/reservation do quản trị viên cấp vẫn không quan sát được. Không sửa/dừng job khác và không khẳng định không có tranh chấp tài nguyên.

Pilot thực tế: {pilot['wall_seconds']:.2f}s wall, GPU wall upper bound {pilot['gpu_hours_upper_bound']:.5f}h; wall gồm cả tải weights nên không phải GPU compute time đo riêng. Peak **PyTorch allocated** {pilot['peak_gpu_allocated_bytes']/1024**2:.2f}MiB; reserved {pilot['peak_gpu_reserved_bytes']/1024**2:.2f}MiB. Không bao gồm CUDA context/cuDNN hoặc job khác. Artifacts {pilot['artifact_bytes']/1024**2:.2f}MiB; dữ liệu/dependencies/weights không nằm trong số này. Model weights public torchvision ImageNet-1K V2 đã lưu và SHA256 trong pilot report.

## STAR feasibility

Đã tải annotation ZIP63.295.542bytes từ author HF release `Zhuzi24/STAR`, revision `7b746149c67c0c0ba36c0309f7344452dcdd2b9c`, verify SHA256; chỉ materialize HDF5/dict/image metadata. Audit: 1.273 ảnh, 48 object/58 predicate classes, HDF5 split0=700 và split2=573, rotated corners trong segmentation arrays, 219.120 objects/401.366 relations. Không áp dụng VG validation5000 vào STAR. [STAR metadata evidence](../reports/P0_STAR_METADATA.json).

Full image multipart archive cần **120.735.953.354bytes =112,44GiB**; filesystem project writable còn **{storage['project_filesystem_available_bytes']/1024**3:.2f}GiB**. Không thấy alternative writable mount đủ chỗ. Chỉ archive+20% reserve đã cần134,93GiB; caches/checkpoints còn phải cộng riêng. Vì vậy chưa tải full STAR images. [Storage audit](../reports/P0_STORAGE_AUDIT.json). Dataset card khai báo Apache-2.0, provenance/quyền ảnh gốc chưa xác minh riêng. Satellite training gate vẫn pending storage, imagery/license verification và author validation policy. Không báo hoàn tất hai miền.

## P1: đóng góp và protocol

[P1 method/preregistration](P1_METHOD_AND_PREREGISTRATION.md) thu hẹp candidate thành relation-conditioned refinement of latent geometric regions; đối chiếu IMP/MSDN/FactorizableNet/BGNN/HybridReciprocal và giữ controls có thể bác bỏ novelty. H1/H2/H3, loss, graph constraints, metrics/checkpoint policy và seed đã cố định trước pilot. Thư viện 31PDF/31MarkItDown vẫn tại [paper index](PAPER_INDEX.md); published results không trộn với reproduced scores. ARR/BRR/HGC chưa implement.

## P2: baseline và kiểm thử

Đã implement VG150/ZIP adapter, directed geometry/union boxes, GC và NGC PredCls R/mR@20/50/100, train-only frequency prior, A0 pair classifier và A1 thêm union feature. A0/A1 cùng frozen ResNet50 ImageNet-1K V2/max-side384, object features/GTclass embeddings/geometry, optimizer/loss/samples. Frontend này chưa phải official VG detector; pilot không là paper baseline reproduction.

**11 golden tests PASS**; official-reference parity trên **32 prediction cases/16 ảnh** (random + trained A1), 12 aggregate metric values sai lệch≤1e-12. Adapter train indices khớp toàn bộ official loader; sampled val boxes/classes/relations khớp reference. Checkpoint reload cho cùng logits chính xác. [Test evidence](../reports/P2_TEST_REPORT.json).

Intentional evaluator differences đã ghi: empty predictions tính recall0 thay vì bị wrapper upstream skip; stable score-tie ordering. Parity scores không có tie tại cutoff. Predicates cùng pair được giữ trong GT, background không được rank, prediction dùng toàn bộ directed nonself pairs; không inject GT-positive pairs khi evaluate. Native box rounding từng gây parity failure đã sửa và thêm regression test trước khi chốt release/pilot.

Pilot seed1729, first eligible64train/32val,10epochs, train256pairs/image≤25%positive, AdamWlr0.001. Chọn checkpoint bằng valGCmR@50; test chưa score. [Fixed config](../configs/p2_pilot.json), [full pilot report](../reports/P2_PILOT_REPORT.json).

| Model | Val GC R@50 (%) | Val GC mR@50 (%) | Selected epoch |
|---|---:|---:|---:|
{chr(10).join(rows)}

A0/A1 train loss giảm; A1 tốt hơn A0 trên subset nhưng frequency vẫn dẫn đầu. **Frequency dùng toàn bộ56.224train annotations, A0/A1 chỉ học64ảnh**, nên đây không phải fair full-training comparison và không thể kết luận phương pháp nào tốt hơn khi converged. Subset deterministic, chưa representative; có validation checkpoint selection, một seed, chưa confidence interval. Không có kết quả SOTA hoặc novelty claim.

Checkpoints tại SSH project `artifacts/A0_pilot.pt`, `artifacts/A1_pilot.pt`; cache/features/frequency counts giữ cùng project. SHA256 checkpoint trong report/status. Source local `src/rsgformer/`, tests/configs/scripts đã sync tới SSH; hướng dẫn chạy lại tại [workspace README](../README.md).

## Gate tiếp theo

P2 natural engineering gate đủ cho bước triển khai ARR có kiểm soát khi người dùng mở P3. Trước confirmatory experiments cần full baseline training/closest controls cùng frontend, ≥3seeds và locked test; trước satellite replication cần storage và release/evaluator/OBB protocol của STAR. Không tự chạy P3–P7 theo phê duyệt P0–P2 hiện tại.
'''
    (ROOT/'research/P0_P1_P2_EXECUTION_REPORT.md').write_text(report,encoding='utf-8')
    (ROOT/'research/P0_P1_P2_STATUS.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    with (ROOT/'reports/P2_REPRODUCED_SCORE_LEDGER.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(['model','task','split','train_images','validation_images','seed','GC_R50_percent','GC_mR50_percent','claim'])
        for k,v in pilot['runs'].items():w.writerow([k,'VG150 PredCls','validation smoke',v.get('train_images_for_prior',64),32,1729,v['metrics']['gc']['R@50']*100,v['metrics']['gc']['mR@50']*100,'smoke_only_not_SOTA'])
    paths=[ROOT/'README.md',ROOT/'.gitignore',ROOT/'data/vg150/split_ids.json',ROOT/'configs/p2_pilot.json',ROOT/'tests/test_protocol.py']
    paths+=list((ROOT/'src').rglob('*.py'))+list((ROOT/'scripts').glob('*.py'))
    paths+=[p for p in (ROOT/'reports').glob('*.json') if p.name!='LOCAL_INTEGRITY_CHECK.json']+list((ROOT/'reports').glob('*.csv'))
    paths += [ROOT/'research'/n for n in ['README.md','RSG_FORMER_RESEARCH_PLAN.md','EXECUTION_AUTHORIZATION.json','P1_METHOD_AND_PREREGISTRATION.md','P0_P1_P2_STATUS.json','P0_P1_P2_EXECUTION_REPORT.md']]
    manifest={'scope':'current P0-P2 source/config/evidence; initial P0 evidence manifest remains immutable','files':[{'file':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(set(paths))]}
    (ROOT/'research/P0_P1_P2_EVIDENCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps({'current_lock':'research/P0_P1_P2_STATUS.json','evidence_files':len(manifest['files']),'status':'written'}))
if __name__=='__main__':main()
