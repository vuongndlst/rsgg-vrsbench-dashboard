"""Fetch only registered failed-parent evidence, byte exact; no remote writes."""
from pathlib import Path,PurePosixPath
import json,sys,hashlib,zipfile,stat,os
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research/tools'))
from p0_audit import connect
report=ROOT/'reports/VG_FOCUS_REPAIR_FAILURE_READONLY_20261030.json';m=json.loads(report.read_text(encoding='utf8'));assert not m['check_access_opened'] and m['parent_processes_absent']
out=ROOT/'outputs/vg_focus_repair_failure_20261030';assert not out.exists();out.mkdir()
archive=ROOT/'artifacts/vg_focus_repair_failure_evidence_20261030/failure_evidence.zip';assert not archive.exists();archive.parent.mkdir(parents=True,exist_ok=True)
c=None;count=total=0
try:
 c=connect();s=c.open_sftp();home=s.normalize('.');base=home+'/rsg-former/artifacts/vg_focus_repair_v1_20261028/'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  z.writestr('manifest.json',report.read_bytes())
  for rel,h in m['parent_file_sha256'].items():
   q=PurePosixPath(rel);assert not q.is_absolute() and '..' not in q.parts
   st=s.lstat(base+rel);assert stat.S_ISREG(st.st_mode) and st.st_size<64*1024**2
   target=out/'parent'/rel;target.parent.mkdir(parents=True,exist_ok=True)
   with s.open(base+rel,'rb') as f,target.open('xb') as dst:
    digest=hashlib.sha256()
    while True:
     block=f.read(1024*1024)
     if not block:break
     total+=len(block);assert total<=64*1024**2;digest.update(block);dst.write(block)
   assert digest.hexdigest()==h,rel
   z.write(target,'parent/'+rel);count+=1
 contact=datetime.now(timezone.utc).isoformat()
finally:
 if c:c.close()
assert total==m['parent_total_bytes']
for rel,h in m['source_sha256'].items():assert hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()==h,rel
parent=out/'parent';lock=json.loads((parent/'preregistration.json').read_text());selection=json.loads((parent/'selection.json').read_text());metrics=json.loads((parent/'train_metrics/complete.json').read_text());gen=json.loads((parent/'train_generation/complete.json').read_text())
assert hashlib.sha256((parent/'preregistration.json').read_bytes()).hexdigest()==m['parent_preregistration_sha256']
assert selection['TRAIN_metrics_completion_sha256']==hashlib.sha256((parent/'train_metrics/complete.json').read_bytes()).hexdigest()
rows=[json.loads(x) for x in (parent/'train_generation/generation_commit.jsonl').read_text().splitlines()]
assert len(rows)==2400 and len({(r['unit_id'],r['policy']) for r in rows})==2400
for policy in ['L0','L1','L2']:assert len([r for r in rows if r['policy']==policy])==800
assert gen['generation_sha256']==hashlib.sha256((parent/'train_generation/generation_commit.jsonl').read_bytes()).hexdigest()
assert metrics['selected_policy']==selection['selected_policy']=='L2'
for policy,h in metrics['metric_sha256'].items():assert hashlib.sha256((parent/'train_metrics'/(policy+'.json')).read_bytes()).hexdigest()==h
result={'status':'PASS_RAW_FAILED_PARENT_SOURCES_TRAIN_GENERATION_METRICS_SELECTION','utc':datetime.now(timezone.utc).isoformat(),'last_successful_remote_contact_utc':contact,'archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'archive_bytes':archive.stat().st_size,'raw_files_verified':count,'raw_bytes':total,'TRAIN_outputs':2400,'selected_from_TRAIN':'L2','preregistration_sha256':m['parent_preregistration_sha256'],'check_access_opened':False,'TEST_accessed':False,'remote_write_performed':False,'metrics_recomputed':False,'resource_snapshot':m['resource_snapshot']}
(ROOT/'reports/VG_FOCUS_REPAIR_FAILURE_VERIFICATION_20261030.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
