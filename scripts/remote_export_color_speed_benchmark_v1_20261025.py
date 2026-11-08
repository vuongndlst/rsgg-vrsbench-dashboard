"""Verify/export completed speed benchmark, no inference and no TEST."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
root=Path.home()/'rsg-former';art=root/'artifacts/vg_color_speed_benchmark_v1_20261025';dest=art/'verified_export_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if (dest/'manifest.json').exists():print((dest/'manifest.json').read_text());raise SystemExit(0)
assert not dest.exists() and shutil.disk_usage(root).free>=16*1024**3+64*1024**2
result=json.loads((art/'complete.json').read_text());lock=json.loads((art/'preregistration.json').read_text())
assert result['status']=='FINITE_TRAIN_COLOR_SPEED_BENCHMARK_COMPLETE' and not result['TEST_accessed'] and not result['VAL_check_reopened']
for p,h in lock['source_sha256'].items():assert sha(root/p)==h
assert sha(art/'membership.json')==lock['membership_sha256']
parent=root/'artifacts/color_rules_v4_train_pilot_20261025';parent_lock=json.loads((parent/'preregistration.json').read_text())
assert sha(parent/'candidate_records.jsonl.gz')==lock['parent_TRAIN_records_sha256']
assert sha(root/'artifacts/color_research_v3_20261025/w2c.mat')==parent_lock['ColorNames_table_sha256']
clip=root.parent/'shared-assets/models/clip-vit-base-patch32';weights=[p for p in clip.iterdir() if p.suffix in ['.bin','.safetensors']]
assert len(weights)==1 and sha(weights[0])=='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f'
dest.mkdir();files={p.name:p for p in art.iterdir() if p.is_file() and p.suffix in ['.json','.gz']}
for p in lock['source_sha256']:files['locked_source/'+p]=root/p
with zipfile.ZipFile(dest/'verified_evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
manifest={'status':'COLOR_SPEED_BENCHMARK_COMPLETION_SOURCE_INPUT_MODEL_VERIFICATION_PASS','utc':datetime.now(timezone.utc).isoformat(),
 'file_sha256':{name:sha(p) for name,p in files.items()},'archive_sha256':sha(dest/'verified_evidence.zip'),'archive_bytes':(dest/'verified_evidence.zip').stat().st_size,
 'TEST_accessed':False,'full_TEST_allowed':False,'visual_quality_gate_passed':False}
(dest/'manifest.json').write_text(json.dumps(manifest,indent=2));(root/'reports/VG_COLOR_SPEED_BENCHMARK_COMPLETION_VERIFICATION_20261025.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest))
