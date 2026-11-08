"""Audit immutable launch/membership/source metadata only; no running color stream reads."""
from pathlib import Path
import json,zipfile
root=Path.home()/'rsg-former';art=root/'artifacts/vg_caption_pilot100_v1_20261026'
from rsgformer.vg_color_execution_common_v1 import sha
assert sha(art/'preregistration.json')=='3715c7489698cae4bdc99f41116f4214ceb5e0fa0b4e772fd57d9123efe06c15'
lock=json.loads((art/'preregistration.json').read_text())
for p,h in lock['source_sha256'].items():assert sha(root/p)==h
m=json.loads((art/'prepare/membership.json').read_text())
assert len(set(m['pilot_VAL_images']))==len(set(m['pilot_unit_ids']))==100
assert len(set(m['TRAIN_bank_images']))==200 and not set(m['pilot_VAL_images'])&set(m['excluded_caption_color_check_images'])
files={'preregistration.json':art/'preregistration.json','membership.json':art/'prepare/membership.json','preparation_preregistration.json':art/'prepare/preregistration.json',
 'target_manifest.json':art/'prepare/target_manifest.json','config.json':root/'configs/vg_caption_pilot100_v1_20261026.json'}
out=root/'reports/VG_CAPTION_PILOT100_INITIAL_EVIDENCE_20261026.zip';assert not out.exists()
with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
 for n,p in files.items():z.write(p,n)
report={'status':'PASS_IMMUTABLE_SOURCE_AND100VAL_MEMBERSHIP_INITIAL_AUDIT','archive':str(out.relative_to(root)),'archive_sha256':sha(out),
 'files_sha256':{n:sha(p) for n,p in files.items()},'reference_text_or_TEST_read':False,'in_progress_inference_records_read':False}
print(json.dumps(report))
