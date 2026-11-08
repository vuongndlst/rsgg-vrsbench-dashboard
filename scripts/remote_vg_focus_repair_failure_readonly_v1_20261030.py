"""Bounded failure preservation; no model, metric, reference, or TEST execution."""
from pathlib import Path
import json,hashlib,os,resource,signal,shutil,zipfile,subprocess
from datetime import datetime,timezone
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2));signal.alarm(75)
root=Path.home()/'rsg-former';parent=root/'artifacts/vg_focus_repair_v1_20261028';out=root/'artifacts/vg_focus_repair_failure_evidence_20261030'
assert not out.exists()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
failure=json.loads((parent/'failure.json').read_text());assert failure['reason']=='GPU_admission' and not failure['check_access_opened']
assert not (parent/'check_access.json').exists() and not (parent/'check_generation').exists() and not (parent/'check_metrics').exists()
assert not (Path('/proc')/str(failure['controller_pid'])).exists() and not (Path('/proc')/str(failure['child_pid'])).exists()
lock=json.loads((parent/'preregistration.json').read_text())
assert sha(parent/'preregistration.json')=='bbfb2e407906c5d6dcd036d3280b49eb3fddb4220789665f3aa10d297f6d0e82'
for p,h in lock['source_sha256'].items():assert sha(root/p)==h,p
files=[p for p in parent.rglob('*') if p.is_file()];assert sum(p.stat().st_size for p in files)<64*1024**2
selection=json.loads((parent/'selection.json').read_text());train=json.loads((parent/'train_metrics/complete.json').read_text())
assert selection['TRAIN_metrics_completion_sha256']==sha(parent/'train_metrics/complete.json') and train['selected_policy']==selection['selected_policy']=='L2'
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'READONLY_FAILURE_EVIDENCE_INDEX','failure':failure,'parent_preregistration_sha256':sha(parent/'preregistration.json'),'source_sha256':lock['source_sha256'],'parent_file_sha256':{p.relative_to(parent).as_posix():sha(p) for p in files},'parent_total_bytes':sum(p.stat().st_size for p in files),'check_access_opened':False,'check_directories_exist':False,'parent_processes_absent':True,'TRAIN_outputs':2400,'selected_from_TRAIN':'L2','TEST_accessed':False,'resource_snapshot':{'disk_free_bytes':shutil.disk_usage(root).free,'GPU_free_MiB':int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0]),'RAM_available_bytes':int(next(x for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')).split()[1])*1024}}
print(json.dumps(result))
