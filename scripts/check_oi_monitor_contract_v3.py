"""Local synthetic status snapshots; no SSH or job modifications."""
from pathlib import Path
from datetime import datetime, timezone
import contextlib, copy, hashlib, importlib.util, io, json, types
root = Path(__file__).absolute().parents[1]
folder = root / 'artifacts/monitor_oi_fixture_v3'
assert not folder.exists()
folder.mkdir(parents=True)
spec = importlib.util.spec_from_file_location('monitor_oi_fixture', root/'scripts/monitor_vg_v2.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
snapshot = json.loads((root/'reports/VG_V2_LAST_STATUS.json').read_text())
base_state = json.loads((root/'reports/VG_V2_MONITOR_STATE.json').read_text())

def run(name, status, repeat=False):
    case = folder/name
    if not repeat:
        (case/'reports').mkdir(parents=True)
        (case/'configs').mkdir()
        (case/'configs/vg_method_v2.json').write_bytes((root/'configs/vg_method_v2.json').read_bytes())
        (case/'reports/VG_V2_MONITOR_STATE.json').write_text(json.dumps(base_state))
    module.ROOT=case
    module.STATE=case/'reports/VG_V2_MONITOR_STATE.json'
    module.LAST=case/'reports/VG_V2_MONITOR_LAST_CHECK.json'
    module.subprocess=types.SimpleNamespace(run=lambda *args,**kwargs:types.SimpleNamespace(returncode=0,stdout=json.dumps(status)))
    with contextlib.redirect_stdout(io.StringIO()): module.main()
    return json.loads(module.LAST.read_text())

value=copy.deepcopy(snapshot)
for process in value['processes'].values(): process['active']=False
value.pop('VG_V2_RESULT.json',None)
value['oi_stream_audit']={'active':True,'pid':123,'progress':{'batches':1,'indexed_images':20}}
value['OI_STREAM_PHASE_LAUNCH.json']={'status':'launched','preregistration_sha256':'synthetic-phase'}
out=run('vg_exit_oi_active',value)
assert out['active'] is False
assert any(event.get('condition')=='unexpected_process_exit' for event in out['events'])

value=copy.deepcopy(snapshot)
value['OI_STREAM_PHASE_RESULT.json']={'status':'incomplete_source_retry_exhausted',
    'preregistration_sha256':'synthetic-phase','last_progress':{'indexed_images':125435}}
first=run('terminal_dedup',value)
assert any(event.get('stage')=='openimages_full_test_pixel_identity_audit' and event['type']=='anomaly' for event in first['events'])
second=run('terminal_dedup',value,repeat=True)
assert not any(event.get('stage')=='openimages_full_test_pixel_identity_audit' for event in second['events'])

value['OI_STREAM_PHASE_RESULT.json']={'status':'pixel_index_complete',
    'preregistration_sha256':'synthetic-phase2','last_progress':{'indexed_images':125436}}
done=run('complete_no_clean_claim',value)
event=next(event for event in done['events'] if event.get('stage')=='openimages_full_test_pixel_identity_audit')
assert event['type']=='pixel_audit_stage_finished' and event['clean_release_certified'] is False and event['test_scored'] is False
value=copy.deepcopy(snapshot)
value['CAPTION_EXACT_PIXEL_COMPARISON.json']={'retained_draft_ids_with_exact_protected_matches':{'train':[2345961,99999999]}}
value['CAPTION_EXACT_PIXEL_COMPARISON_V2.json']={'retained_draft_ids_with_exact_protected_matches':{'train':[2345961,99999999]}}
first=run('cross_report_exact_dedup',value)
matches=[event for event in first['events'] if event.get('condition')=='exact_protected_pixels_retained_in_provisional_draft']
assert len(matches)==1 and matches[0]['vg_id']==99999999
second=run('cross_report_exact_dedup',value,repeat=True)
assert not any(event.get('condition')=='exact_protected_pixels_retained_in_provisional_draft' for event in second['events'])
value=copy.deepcopy(snapshot)
value['OI_STREAM_PHASE_LAUNCH.json']={'status':'launched','preregistration_sha256':'synthetic-retry-phase'}
value['OI_STREAM_FINGERPRINT_PROGRESS.json']['retry_exhausted_images']=2
first=run('retry_exhaustion_condition_dedup',value)
assert sum(event.get('condition')=='oi_test_pixel_sources_exhausted_registered_attempts' for event in first['events'])==1
value['OI_STREAM_FINGERPRINT_PROGRESS.json']['retry_exhausted_images']=3
second=run('retry_exhaustion_condition_dedup',value,repeat=True)
assert not any(event.get('condition')=='oi_test_pixel_sources_exhausted_registered_attempts' for event in second['events'])
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'cases':['OI activity cannot hide VG exit','incomplete phase stays anomaly','terminal event durable dedup','complete index is not clean/test score','same exact ID across v1/v2 reports emits once and durable dedup','retry-exhaustion condition emits once per phase despite later ID count changes'],
        'monitor_sha256':hashlib.sha256((root/'scripts/monitor_vg_v2.py').read_bytes()).hexdigest(),
        'ssh_calls':0,'real_monitor_state_modified':False,'real_jobs_modified':False}
(root/'reports/OI_MONITOR_CONTRACT_PROBE_V3.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
