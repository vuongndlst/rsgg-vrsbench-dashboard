# -*- coding: utf-8 -*-
"""Local, reproducible tables from the completed train/val audit; no model/test access."""
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'reports/VG150_OBJECT_COLOR_TRAIN_VAL_AUDIT_V2_20261024.json'
data = json.loads(SOURCE.read_text(encoding='utf-8-sig'))
assert data['status'] == 'VG150_TRAIN_VAL_OBJECT_COLOR_AUDIT_V2_COMPLETE'
assert not data['test_attribute_semantics_analyzed'] and not data['model_loaded']
basic = ['black', 'white', 'gray', 'red', 'green', 'blue', 'yellow', 'brown', 'orange', 'pink', 'purple']
pilot = ['shirt', 'jacket', 'pant', 'short', 'coat', 'hat', 'cap', 'helmet', 'shoe', 'bag',
         'bus', 'truck', 'car', 'train', 'cat', 'dog', 'horse', 'cow', 'plate', 'toilet']
val = {r['class_id']: r for r in data['splits']['val']}
rows = []
for rank, tr in enumerate(sorted(data['splits']['train'], key=lambda r: (-r['color_rate_all_objects'], r['class_id'])), 1):
    va = val[tr['class_id']]
    rows.append(dict(rank_train_rate=rank, class_id=tr['class_id'], class_name=tr['class_name'],
        train_objects=tr['objects'], train_objects_with_color=tr['with_color'],
        train_color_percent=round(100*tr['color_rate_all_objects'], 4),
        train_attribute_annotated_objects=tr['with_any_attribute'],
        train_color_percent_among_attribute_annotated=round(100*tr['color_rate_among_attribute_annotated'], 4),
        train_multicolor_objects=tr['with_multiple_color_labels'],
        train_explicit_multicolored=tr['with_multicolored_attribute'],
        train_explicit_colorful=tr['with_colorful_attribute'],
        train_label_histogram_entropy_bits=tr['color_label_entropy_bits'],
        val_objects=va['objects'], val_objects_with_color=va['with_color'],
        val_color_percent=round(100*va['color_rate_all_objects'], 4),
        top3_train_colors='; '.join(f'{c}:{n}' for c,n in tr['top_colors'][:3]),
        pilot_candidate=tr['class_name'] in pilot,
        note='annotation frequency; absent color is unlabeled; pilot status is a proposal'))
assert len(rows) == 150 and len({r['class_id'] for r in rows}) == 150
csvpath = ROOT/'reports/VG150_COLOR_PRIORITY_20261024.csv'
with csvpath.open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
totals={}
for split, values in data['splits'].items():
    hist=Counter()
    for r in values: hist.update(r['colors'])
    n=sum(r['objects'] for r in values); k=sum(r['with_color'] for r in values)
    totals[split]=dict(objects=n, with_color=k, color_percent=100*k/n,
        with_any_attribute=sum(r['with_any_attribute'] for r in values),
        multicolor_objects=sum(r['with_multiple_color_labels'] for r in values),
        explicit_multicolored=sum(r['with_multicolored_attribute'] for r in values),
        explicit_colorful=sum(r['with_colorful_attribute'] for r in values),
        normalized_color_label_histogram=dict(hist.most_common()),
        basic11_share_of_label_occurrences=sum(hist[c] for c in basic)/sum(hist.values()))
summary=dict(status='TRAIN_VAL_COLOR_PRIORITY_TABLES_COMPLETE_PLAN_ONLY',
    updated_utc=datetime.now(timezone.utc).isoformat(), source=str(SOURCE.relative_to(ROOT)),
    source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    audit_version=2, all_150_classes_present=True,
    original_first_audit_preserved=True,
    palette_correction='V2 includes literal maroon/cream/khaki omitted by initial lexical audit; separate multicolored/colorful counts',
    totals=totals, initial_palette=basic,
    extended_palette=['tan','beige','maroon','cream','khaki'],
    appearance_extensions=['silver','gold'], hair_only_extension=['blond'],
    palette_reduction_is_not_used_to_relabel_original_gt=True,
    proposed_pilot_classes=pilot,
    pilot_class_selection='TRAIN frequency/support plus semantic suitability; plate/toilet as low-variation controls; not a validated production whitelist',
    caveats=['Object-bound weak H5 attribute annotations, not pixel-verified color truth',
        'All canonical in-scope H5 object instances; no relation-active mask filtering',
        'Absent color annotation is unlabeled, not a negative',
        '200-attribute vocabulary is incomplete; no raw attributes.json audit claimed',
        'Histogram occurrence coverage is not object-level coverage or accuracy',
        'Multi-label colors can make histogram sum exceed colored-object count',
        'Object-binomial Wilson bounds in source are descriptive only; future intervals cluster by image'],
    test_attributes_analyzed=False, model_loaded=False, color_extraction_run=False,
    caption_generated=False, caption_test_scored=False,
    artifacts={'csv':str(csvpath.relative_to(ROOT)), 'markdown':'reports/VG150_COLOR_PRIORITY_20261024.md'})
md=['# Màu được ghi trong GT theo đủ 150 lớp VG150 — TRAIN/VAL', '',
    'Nguồn: `VG150_OBJECT_COLOR_TRAIN_VAL_AUDIT_V2_20261024.json`. Xếp hạng theo TRAIN; VAL trình bày riêng. Không phân tích thuộc tính TEST.', '',
    '**Tỷ lệ = số object có ít nhất một nhãn màu / toàn bộ object của lớp.** Đây là tần suất được gán nhãn, không phải xác suất object có màu. Thiếu attribute là chưa được ghi nhãn. Bộ H5 chỉ giữ 200 attribute, không đại diện mọi nhãn trong VG gốc.', '',
    'V2 bổ sung maroon/cream/khaki; giữ audit lexical ban đầu để truy vết. `multicolored` và `colorful` đếm riêng, không tự quy đổi thành màu cụ thể. Tổng histogram là số lần xuất hiện nhãn, không phải số object.', '',
    '| Split | Objects | Có nhãn màu | Tỷ lệ | >=2 màu | 11 màu cơ bản / tổng lần xuất hiện nhãn |',
    '|---|---:|---:|---:|---:|---:|']
for s,t in totals.items():md.append(f"| {s} | {t['objects']} | {t['with_color']} | {t['color_percent']:.2f}% | {t['multicolor_objects']} | {100*t['basic11_share_of_label_occurrences']:.2f}% |")
md+=['', '11 màu cơ bản: '+', '.join(basic)+'. Màu mở rộng: tan, beige, maroon, cream, khaki. Silver/gold cần phân biệt màu và bề ngoài kim loại; blond dành riêng cho tóc.', '',
    'Dấu * chỉ nhóm pilot đề xuất, chưa phải whitelist đã kiểm định. Chỉ số entropy nhãn và mẫu số theo attribute có trong CSV; không dùng tỷ lệ này để đoán màu từ tên lớp.', '',
    '| Hạng TRAIN | Lớp | TRAIN có màu / tổng | TRAIN % | VAL có màu / tổng | VAL % | Ba màu TRAIN thường gặp | Pilot |',
    '|---:|---|---:|---:|---:|---:|---|:---:|']
for r in rows:
    md.append(f"| {r['rank_train_rate']} | {r['class_name']} | {r['train_objects_with_color']}/{r['train_objects']} | {r['train_color_percent']:.2f} | {r['val_objects_with_color']}/{r['val_objects']} | {r['val_color_percent']:.2f} | {r['top3_train_colors']} | {'*' if r['pilot_candidate'] else ''} |")
md+=['', '## Diễn giải trước khi chọn lớp', '',
    '- Quần áo/phụ kiện và xe: tần suất cao và màu có ích để phân biệt instance; phải kiểm tra mask, kích thước và màu logo/nền.',
    '- Động vật: giữ đa màu khi có bằng chứng; không ép một màu duy nhất.',
    '- Toilet/plate có tần suất cao nhưng màu thường ít biến thiên; dùng làm đối chứng thiên lệch prior, không tự gán white.',
    '- Person/man/woman: không gán màu quần áo/da/tóc cho toàn người. Gắn màu vào node shirt/hat/hair tương ứng; không suy ra sắc tộc.',
    '- Đủ 150 lớp được báo cáo; lớp thiếu mẫu validation không được coi là đã kiểm định. Mọi instance lỗi/unknown giữ trong mẫu số phù hợp.', '']
(ROOT/summary['artifacts']['markdown']).write_text('\n'.join(md),encoding='utf-8')
(ROOT/'reports/VG_COLOR_AUDIT_SUMMARY_20261024.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'status':summary['status'],'classes':len(rows),'totals':totals,'csv':str(csvpath)},ensure_ascii=False))
