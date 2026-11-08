"""Explicit user amendment: automated VAL quality assessment, no human labels invented."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,resource,shutil
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_auto_val_diagnostic_v1_20261025'
assert not ART.exists() and shutil.disk_usage(ROOT).free>=16*1024**3+128*1024**2
os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[0]})
from rsgformer.vg_color_attribute_diagnostic_v1 import evaluate
parent=ROOT/'artifacts/color_fresh_val_execution_r1_20261025';complete=json.loads((parent/'complete.json').read_text())
assert complete['nodes']==3083 and hashlib.sha256((parent/'candidate_records.jsonl.gz').read_bytes()).hexdigest()==complete['candidate_records_sha256']
ART.mkdir();lock={'utc':datetime.now(timezone.utc).isoformat(),'human_authorization':'25Oct chose automated evaluation with explicit limitations, then exploratory full colorTEST',
 'no_human_precision_claim':True,'TEST_access_allowed':False,'config_unchanged':True,'selected_policy':'SAM2 conservative + native binding strict, safe_cache exact parity',
 'source_sha256':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['scripts/remote_vg_color_auto_val_diagnostic_v1_20261025.py','src/rsgformer/vg_color_attribute_diagnostic_v1.py']},
 'candidate_records_sha256':complete['candidate_records_sha256'],'quality_statistics':'descriptive, no post-hoc quality threshold or tuning from check'}
(ART/'preregistration.json').write_text(json.dumps(lock,indent=2))
membership=json.loads((ROOT/'artifacts/color_fresh_val_manifest_v1_20261025/membership.json').read_text())
result=evaluate(ROOT,parent/'candidate_records.jsonl.gz',ART/'diagnostic',sum(membership.values(),[]),parent/'target_manifest.json','val')
assert result['counts']['selected_nodes']==3083 and resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3
result['utc']=datetime.now(timezone.utc).isoformat();result['TEST_accessed']=False
(ART/'complete.json').write_text(json.dumps(result,indent=2));(ROOT/'reports/VG_COLOR_AUTO_VAL_DIAGNOSTIC_V1_20261025.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result))
