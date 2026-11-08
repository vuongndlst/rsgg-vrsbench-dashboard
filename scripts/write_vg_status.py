"""Write a concise snapshot of the ongoing VG experiment, without inventing results."""
from pathlib import Path
import json,datetime
ROOT=Path(__file__).absolute().parents[1]
def main():
    status=json.loads((ROOT/'reports/VG_METHOD_STATUS.json').read_text());prereg=json.loads((ROOT/'reports/VG_METHOD_PREREGISTRATION.json').read_text())
    probe=json.loads((ROOT/'reports/VG_METHOD_CONTRACT_PROBE.json').read_text());cache=status.get('cache',{});progress=status.get('VG_METHOD_PROGRESS.json',{})
    images=cache.get('images',{});outcome=status.get('VG_METHOD_RESULT.json',{})
    interpretation={
        'controlled_test_criteria_met':'Các tiêu chí preregistered đã đạt trên locked test trong protocol VG PredCls này; chưa phải SOTA hoặc chứng minh mọi frontend/domain.',
        'validation_criteria_met':'Validation đã đạt tiêu chí; cần kiểm tra tiến độ locked test, chưa coi đây là independent-test confirmation.',
        'test_did_not_meet_criteria':'Locked test không đạt tiêu chí; không claim superiority.',
        'confirmation_did_not_meet_criteria':'Ba-seed validation confirmation không đạt tiêu chí; không mở test hoặc claim superiority.',
        'screening_failed_to_establish_benefit':'Screening chưa thiết lập lợi ích; dừng trước confirmation. Không suy ra hiệu ứng bằng0 khi chưa converged.',
        'stopped_on_error':'Pipeline đã dừng do lỗi/cap; xem reason trong JSON. Không có kết luận khoa học từ lỗi kỹ thuật.'
    }.get(outcome.get('status'),'Chưa có full-validation/test superiority result; experiment đang chạy.')
    text=f'''# VG method — trạng thái thực thi

Snapshot lấy từ SSH; progress timestamp `{progress.get('utc','unknown')}`. **Process active: {status.get('process_active',False)}**. Stage: `{progress.get('event','unknown')}`. Đây là trạng thái chạy, chưa phải bằng chứng method tốt hơn.

Người dùng đã trực tiếp mở thực nghiệm chứng minh method trên VG trước Open Images/STAR. Pipeline chạy tại `${{REMOTE_HOME}}/rsg-former`, không phụ thuộc việc cửa sổ SSH/local runner còn mở. Hiện cache train **{images.get('train',0):,}/56.224** ảnh, validation **{images.get('val',0):,}/5.000**; bytes cache `{cache.get('bytes',0)}`. Ảnh VG giữ nguyên ZIP.

Đã implement ARR/BRR, fixed/one-way controls, feature+membership capacity control và GNN. 7mechanism tests PASS, 6train-only optimizer/checkpoint probes PASS; pipeline cache shard thực tế đã kiểm tra ID/positive pairs/schema/source hashes. Registered parameters2.139.351/model, gradient-active count được báo riêng theo variant. Full và feature_membership control dùng cùng params/geometry head/features, chỉ full áp delta vào region box logits.

Sequence tự thực thi hữu hạn:

1. Tạo frozen ResNet50 features640/stride16 cho fulltrain/fullval; khoảng19GiB dự kiến, thêm metadata/checkpoints vẫn phải dưới32GiB artifact cap.
2. Fulltrain/fullval screening6variants, seed1729,5epochs/model. Nếu full không vượt one_way và feature_membership với R guardrail, dừng và lưu insufficient evidence.
3. Nếu screening đạt, train lại từ đầu5variants×3seeds×20epochs. Primary: gain≥1ppGCmR@50 so capacity control, R@50 drop≤1pp, paired-image bootstrap lower95%CI>0; báo mọi control và seed variance.
4. Nếu validation criteria đạt, khóa checkpoint/config/split trước khi mở full26.446test. Stream cùng features một lần cho15models; không dùng test tune/chọn epoch. Test không đạt thì không claim superiority.

{interpretation} Outcome snapshot: `{outcome.get('status','running; no final outcome yet')}`. Probe không score validation/test. P2 smoke64train/32val là evidence lịch sử, không làm positive scientific result cho run này. Nếu result mới đã tồn tại, xem đầy đủ statistics/reason trong server `reports/VG_METHOD_RESULT.json`; source/model claims luôn theo protocol đã khóa.

Budget bảo vệ: allocator8GiB (không phải total-process VRAM), CPU4threads, artifacts≤32GiB, free disk reserve16GiB, aggregatewall≤80h, screening≤12h, mỗi run≤8h. GPU dùng chung; không suy ra entitlement độc quyền. Train-only probe frontend64images mất{probe['feature_seconds']:.2f}s; extrapolation cache khoảng{probe['projected_cache_wall_hours']:.2f}h chỉ là estimate, warm-up/shared GPU có thể làm sai ETA. Đọc tiến độ thật thay vì dùng estimate như completion time.

Source/config lock: [remote preregistration snapshot](../reports/VG_METHOD_PREREGISTRATION.json), [protocol](VG_METHOD_PREREGISTRATION_V1.md), [config](../configs/vg_method_v1.json), [local integrity](../reports/VG_METHOD_LOCAL_INTEGRITY.json), [cache integrity](../reports/VG_METHOD_CACHE_INTEGRITY.json). Source mới đang khóa; không chỉnh giữa run. Source/hash manifest P0–P2 là lịch sử trước ARR/BRR, không phải hash current source.

Status local [VG_METHOD_STATUS.json](../reports/VG_METHOD_STATUS.json) chỉ cập nhật khi fetch. Logs/progress/checkpoints tiếp tục ghi trên SSH. Đọc snapshot mới từ PowerShell:

```powershell
python scripts/ssh_project.py run scripts/remote_vg_status.py --environment project
python scripts/ssh_project.py fetch reports/VG_METHOD_STATUS.json
python scripts/write_vg_status.py
```

Trên SSH: logs `~/rsg-former/artifacts/vg_method_v1/pipeline.log`; final report `~/rsg-former/reports/VG_METHOD_RESULT.json`, screening/confirmation/locked-test reports cùng thư mục. STOP flag trong experiment artifact directory dừng ở guard kế tiếp. Resume cùng source/config/budget, không mở lại test âm thầm nếu đã score. Chưa train Open Images/STAR, chưa triển khai HGC/SGDet, chưa có SOTA claim.
'''
    (ROOT/'research/VG_METHOD_STATUS.md').write_text(text,encoding='utf-8');print(json.dumps({'status':'snapshot_written','process_active':status.get('process_active',False),'train_cache_images':images.get('train',0)}))
if __name__=='__main__':main()
