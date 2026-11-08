"""Prepare a new train/validation-only phase, without resuming stopped jobs."""
from pathlib import Path
from datetime import datetime,timezone
import ast,copy,hashlib,json
ROOT=Path(__file__).absolute().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(rel,obj):
 p=ROOT/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf8')
parent_path=ROOT/'reports/VG_TIME_EXTENSION_V1_PREREGISTRATION.json';parent=json.loads(parent_path.read_text(encoding='utf8'))
remote_parent_sha=hashlib.sha256(parent_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest()
assert remote_parent_sha=='7e7a41ca5a0943e75534c80a18ada65138ad6657f6424f30ce605a7e71b3371e'
original=ROOT/'scripts/remote_vg_v2_time_extension_v1.py';assert sha(original)=='17dd07c012db46b8e17ad42aadef03a1179a42b6ce9ab036fd0a9f3e897c3a44'
cfg=copy.deepcopy(parent['config']);cfg.update(execution_revision='vg_context_seed_replication_v1',replication_variant='context',replication_seeds=[2718,3141],replication_purpose='Human24Oct requested additional seeds of the validation-selected Context component. Fixed recipe replication after prior exploratory single-seed test; train/validation only. No tuning from test observations and no fresh test claim.',test_policy='No TEST dataset/loader/evaluator or test access in this phase; previous Context1729 test result remains frozen.',new_phase_artifact_limit_gib=2)
save('configs/vg_context_seed_replication_v1.json',cfg)
source=original.read_text(encoding='utf8').split(" screen={v:run(",1)[0]
def replace_once(a,b):
 global source
 assert source.count(a)==1,a
 source=source.replace(a,b)
replace_once("ART=ROOT/'artifacts/vg_method_v2_speed_m2_v1'","ART=ROOT/'artifacts/vg_context_seed_replication_v1'")
replace_once("ROOT/'configs/vg_method_v2_time_extension_v1.json'","ROOT/'configs/vg_context_seed_replication_v1.json'")
replace_once("ART/'time_extension_v1/preregistration.json'","ART/'preregistration.json'")
source=source.replace('VG_V2_TIME_EXT_PROGRESS.json','VG_CONTEXT_SEED_REPLICATION_V1_PROGRESS.json').replace('VG_V2_TIME_EXT_RESULT.json','VG_CONTEXT_SEED_REPLICATION_V1_RESULT.json')
source=source.replace(" test_manifest={'cache_dtype':'float32','splits':{'test':[]},'status':'building'}\n",'').replace(" test_cache=ART/'features_test'\n",'')
replace_once("  entries=list((test_manifest if split=='test' else manifest)['splits'][split])","  assert split in ('train','val'),'test_access_forbidden'\n  entries=list(manifest['splits'][split])")
replace_once("  ds=train if split=='train' else val if split=='val' else test","  ds=train if split=='train' else val")
replace_once("p=(test_cache if split=='test' else cache)/e['path']","p=cache/e['path']")
source=source.replace("'validation' if split=='val' else 'locked_test'","'validation'")
begin=source.index(' def feature_cache(');end=source.index(' def run(',begin);source=source[:begin]+source[end:]
replace_once("ART/'time_extension_v1/initialization.json'","ART/'initialization.json'")
replace_once("for base in [ROOT/'artifacts/vg_method_v2',ART]","for base in [ROOT/'artifacts/vg_method_v2',ROOT/'artifacts/vg_method_v2_speed_m2_v1',ROOT/'artifacts/vg_component_control_test_v1',ART]")
replace_once(" assert total<=40*1024**3,'aggregate_artifact_budget_reached'"," assert total<=40*1024**3,'aggregate_artifact_budget_reached'\n own=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file())\n assert own<=2*1024**3,'new_phase_artifact_cap'")
replace_once(" initial_sha=initial_digest.hexdigest()"," initial_sha=initial_digest.hexdigest()\n assert initial_sha=='bc2ceb642f57f26729b104da880f9e5057c3b92fac891e615515884d66efe09f','original_initialization_changed'")
source+=''' results={str(seed):run('replication','context',seed,13) for seed in [2718,3141]}
 inherited=json.loads((ROOT/'artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/report.json').read_text())
 assert inherited['status']=='complete' and inherited['checkpoint_sha256']=='29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e'
 all_rows={'1729':inherited,**results};summary={}
 for name in ['R@20','mR@20','R@50','mR@50','R@100','mR@100']:
  values=[row['selected_metrics'][name] for row in all_rows.values()]
  summary[name]={'by_seed':dict(zip(all_rows,values)),'mean':float(np.mean(values)),'sample_std':float(np.std(values,ddof=1))}
 result={'status':'CONTEXT_THREE_SEED_VALIDATION_COMPLETE','variant':'context','new_seeds':[2718,3141],'validation':summary,'runs':all_rows,'test_scored':False,'test_accessed':False,'claim':'Fixed-recipe validation replication only. Prior exploratory single-seed test is frozen; no regional superiority or fresh held-out test claim.'}
 atomic(ROOT/'reports/VG_CONTEXT_SEED_REPLICATION_V1_RESULT.json',result);event('stage_complete',outcome=result['status'],test_scored=False)
if __name__=='__main__':
 try:main()
 except Exception as e:
  out={'status':'stopped_on_error','error_type':type(e).__name__,'reason':str(e)[:800],'traceback':traceback.format_exc()[-5000:],'test_scored':False,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
  atomic(ROOT/'reports/VG_CONTEXT_SEED_REPLICATION_V1_RESULT.json',out);event('stopped_on_error',error_type=out['error_type'],reason=out['reason']);raise
'''
source=source.replace('"""New four-arm matched microbatch2 experiment; cached-head adaptation only."""','"""Human24Oct: fixed Context seeds2718/3141; train/validation-only replication."""')
path=ROOT/'scripts/remote_context_seed_replication_v1.py';path.write_text(source,encoding='utf8');ast.parse(source)
assert "datasets(ROOT,'test')" not in source and 'feature_cache(' not in source and 'test_cache' not in source and 'test_manifest' not in source
assert "run('replication','context',seed,13) for seed in [2718,3141]" in source
preflight=json.loads((ROOT/'reports/VG_CONTEXT_SEED_CONTINUATION_PREFLIGHT_20261024.json').read_text(encoding='utf8'))
locked=copy.deepcopy(parent);locked.update(utc=datetime.now(timezone.utc).isoformat(),config=cfg,execution_revision=cfg['execution_revision'],parent_preregistration_sha256=remote_parent_sha,authorization='Direct human request24Oct2026: train additional seeds of the current trained model. Original STOP and source/config/evidence remain untouched.',test_access_forbidden=True,post_exploratory_test_replication=True,launch_status='NOT_LAUNCHED_RESOURCE_BLOCKED',resource_preflight=preflight['resources'])
locked['source_hashes'].update({p:sha(ROOT/p) for p in ['scripts/remote_context_seed_replication_v1.py','configs/vg_context_seed_replication_v1.json']})
locked['source_hashes']['artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/report.json']=next(x['report_sha256'] for x in preflight['runs'] if x['folder']=='confirmation_context_1729')
save('reports/VG_CONTEXT_SEED_REPLICATION_V1_PREREGISTRATION_DRAFT.json',locked)
save('reports/VG_CONTEXT_SEED_REPLICATION_V1_PREPARATION.json',{'status':'PREPARED_NOT_LAUNCHED_DISK_RESERVE_BLOCKED','variant':'context','new_seeds':[2718,3141],'epochs_per_seed':13,'fresh_original_initialization':True,'reuse_seed1729_completed_evidence_only':True,'test_access_forbidden':True,'original_owned_STOP_preserved':True,'new_runner_ast':'PASS','runner_sha256':sha(path),'resource_preflight':preflight['resources'],'remote_files_written':0})
print(json.dumps({'status':'PREPARED_NOT_LAUNCHED_DISK_RESERVE_BLOCKED','seeds':[2718,3141],'epochs':13,'worker_sha256':sha(path),'test_access_forbidden':True,'disk_free_gib':preflight['resources']['disk_free_bytes']/1024**3,'launch_minimum_free_gib':18,'additional_free_gib_required':18-preflight['resources']['disk_free_bytes']/1024**3}))
