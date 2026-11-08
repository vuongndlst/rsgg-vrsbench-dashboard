"""Byte/protocol verification of completed evidence only; no inference or metric rerun."""
from pathlib import Path,PurePosixPath
from datetime import datetime,timezone
import json,zipfile,hashlib,stat,collections,math
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'reports';archive=ROOT/'artifacts/vg_focus_repair_r1_20261030/focus_repair_r1_results.zip';out=ROOT/'outputs/vg_focus_repair_r1_20261101';parent=ROOT/'outputs/vg_focus_repair_failure_20261030/parent'
status=json.loads((R/'VG_FOCUS_REPAIR_R1_LAST_STATUS_20261030.json').read_text(encoding='utf8'));completion=status['complete.json'];launch=json.loads((R/'VG_FOCUS_REPAIR_R1_LAUNCH_20261030.json').read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
assert sha(archive)==completion['archive_sha256']=='218098eccf4ef9c803be72e3578eecf21082e8787cfbb56bf18bd4eb9eb9bb31'
assert completion['status']=='FOCUS_REPAIR_R1_REUSED_TRAIN_AND_FROZEN_CHECK_COMPLETE' and not completion['TEST_accessed'] and not completion['policy_adopted']
assert not status['controller_pid_proc_exists'];assert not out.exists();out.mkdir()
with zipfile.ZipFile(archive) as z:
 assert sum(i.file_size for i in z.infolist())<64*1024**2
 for i in z.infolist():
  name=PurePosixPath(i.filename);assert not name.is_absolute() and '..' not in name.parts and not stat.S_ISLNK(i.external_attr>>16)
  target=out/i.filename;assert target.resolve().is_relative_to(out.resolve())
 z.extractall(out)
lock=read(out/'preregistration.json');assert sha(out/'preregistration.json')==launch['preregistration_sha256']==completion['preregistration_sha256']=='789d621626a330004bf566924afe4b977fc0dc375567e2f9c5cc10897f2af5b6'
for path,h in lock['source_sha256'].items():assert sha(ROOT/path)==h,path
assert sha(ROOT/'configs/vg_focus_repair_r1_20261030.json')==lock['config_sha256']
assert sha(parent/'preregistration.json')==lock['parent_preregistration_sha256']==completion['parent_preregistration_sha256']
for path,h in lock['parent_file_sha256'].items():assert sha(parent/path)==h,path
assert sha(out/'selection.json')==sha(parent/'selection.json')==completion['selection_sha256']==lock['parent_selection_sha256']
selection=read(out/'selection.json');assert selection['selected_policy']=='L2' and not selection['selection_uses_check_or_TEST'] and not selection['policy_adopted']
assert sha(parent/'train_metrics/complete.json')==selection['TRAIN_metrics_completion_sha256']==completion['TRAIN_completion_sha256']
assert lock['TRAIN_regeneration_allowed'] is False and lock['check_previously_opened'] is False and not lock['selection_changed'] and not lock['check_policy_selection_allowed']
parentrows=[json.loads(x) for x in (parent/'train_generation/generation_commit.jsonl').read_text().splitlines()];assert len(parentrows)==len({(r['policy'],r['unit_id']) for r in parentrows})==2400
access=read(out/'check_access.json');assert access['selected_from_TRAIN_only']=='L2' and access['selection_sha256']==sha(out/'selection.json') and not access['TEST_accessed']
gen=read(out/'check_generation/complete.json');metric=read(out/'check_metrics/complete.json');stages=read(out/'stages_complete.json')
assert gen['outputs']==512 and gen['units_per_policy']==metric['units_per_policy']==256 and gen['policies']==metric['policies']==['L0','L2']
assert sha(out/'check_metrics/complete.json')==completion['check_completion_sha256']
assert sha(out/'check_generation/generation_commit.jsonl')==gen['generation_sha256']==metric['generation_sha256']
assert sha(out/'check_generation/query_units_by_policy.json')==gen['query_units_sha256']
queries=read(out/'check_generation/query_units_by_policy.json');rows=[json.loads(x) for x in (out/'check_generation/generation_commit.jsonl').read_text().splitlines()];refs=read(out/'check_metrics/original_human_references.json')
assert len(rows)==len({(r['policy'],r['unit_id']) for r in rows})==512
unitids={u['unit_id'] for u in queries['L0']};assert len(unitids)==len(refs)==256 and unitids==set(refs)
checkparents={u['parent_image_id'] for u in queries['L0']};assert len(checkparents)==64 and set(collections.Counter(u['parent_image_id'] for u in queries['L0']).values())=={4}
assert not checkparents & {r['parent_image_id'] for r in parentrows}
for policy in ['L0','L2']:
 units=queries[policy];assert len(units)==256 and {u['unit_id'] for u in units}==unitids
 q={u['unit_id']:u for u in units};local=[r for r in rows if r['policy']==policy];assert len(local)==256 and {r['unit_id'] for r in local}==unitids
 for r in local:
  u=q[r['unit_id']];g=u['graph'];assert len(g['nodes'])<=24 and len(g['edges'])<=32 and r['query_graph_sha256']==digest(g)
  assert r['input_tokens']<=768 and len(r['new_token_ids'])<=32 and not any(k in u for k in ['reference','caption','phrase','GT_attributes'])
  assert all(not n.get('candidate_colors') and not n.get('accepted_colors') for n in g['nodes'])
  assert r['empty_evidence']==(not bool(g['nodes']))
  if r['empty_evidence']:assert r['caption']==''
 m=read(out/'check_metrics'/(policy+'.json'));assert sha(out/'check_metrics'/(policy+'.json'))==metric['metric_sha256'][policy]
 assert m['expected_units']==m['evaluated_units']==len(m['unit_ids'])==len(m['per_unit'])==256 and set(m['unit_ids'])==unitids
 assert m['empty_predictions']==sum(not r['caption'].strip() for r in local)==metric['diagnostics'][policy]['empty_predictions']
 assert m['empty_evidence_units']==sum(r['empty_evidence'] for r in local)==metric['diagnostics'][policy]['empty_evidence_units']
 assert m['aggregate_native']==metric['native_scores'][policy] and m['aggregate_display_x100']==metric['display_scores_x100'][policy]
 assert all(math.isfinite(float(v)) for v in m['aggregate_native'].values())
 assert m['SPICE_status']=='NOT_RUN_FOCUS_DIAGNOSTIC'
 assert m['factuality_audit_status']=='AUTOMATED_PROXY_NOT_HUMAN_ACCURACY'
for outcome in completion['outcomes']:
 assert outcome['status']=='COMPLETE' and outcome['returncode']==0 and outcome['guard_reason'] is None
 assert outcome['seconds']<=outcome['wall_seconds']
 assert outcome['peak_owned_tree_RSS_bytes']<=outcome['own_tree_RSS_cap_GiB']*1024**3 and outcome['peak_owned_GPU_bytes']<=outcome['own_GPU_cap_GiB']*1024**3
packet=read(out/'check_metrics/fixed20_factuality_review_packet.json');assert len(packet)==20 and {x['unit_id'] for x in packet}==set(read(parent/'prepare/fixed_check_qualitative_sample_ids.json'))
assert not any(x['human_review_completed'] for x in packet)
assert metric['policy_adopted'] is False and not metric['semantic_correctness_established'] and not metric['automatic_TEST_launch']
assert abs(metric['check_METEOR_difference_native']-(metric['native_scores']['L2']['METEOR']-metric['native_scores']['L0']['METEOR']))<1e-15
result={'status':'PASS_R1_ARCHIVE_SOURCE_PARENT_REUSE_SELECTION_AND512_CHECK_OUTPUTS','utc':datetime.now(timezone.utc).isoformat(),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'source_lock_sha256':sha(out/'preregistration.json'),'parent_preregistration_sha256':lock['parent_preregistration_sha256'],'parent_files_verified':len(lock['parent_file_sha256']),'sources_verified':len(lock['source_sha256']),'TRAIN_outputs_reused':2400,'check_outputs':512,'check_images':64,'check_regions_per_policy':256,'selected_policy':'L2','selected_on_TRAIN_only':True,'new_policy_adopted':False,'SPICE':'NOT_RUN','human_factuality_review_complete':False,'TEST_accessed':False,'metric_recomputed':False,'local_output_directory':out.relative_to(ROOT).as_posix()}
(R/'VG_FOCUS_REPAIR_R1_COMPLETION_VERIFICATION_20261101.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
