"""One owned continuation: wait local backup, exact retirement, admission, exclusive R1."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor
import json,hashlib,os,sys,time,subprocess,traceback,struct
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports';ART=ROOT/'artifacts/maintenance/vg_v2_offload_continuation_20261101';STATUS=REPORTS/'VG_V2_OFFLOAD_CONTINUATION_STATUS_20261101.json'
assert not ART.exists(),'no_duplicate_continuation';ART.mkdir(parents=True)
START=time.monotonic();HISTORY=[]
def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def save(p,d):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2),encoding='utf8');os.replace(temp,p)
def guard():
 assert time.time()<1794670321.2414355 and time.monotonic()-START<9000,'absolute_or_continuation_wall_limit'
 assert not (ART/'STOP').exists(),'owned_continuation_STOP'
def state(stage,**kw):
 d={'status':'RUNNING','stage':stage,'utc':now(),'pid':os.getpid(),'elapsed_seconds':time.monotonic()-START,'history':HISTORY,'TEST_access_allowed':False,**kw};save(STATUS,d);return d
def run(action,path,report=None,timeout=300):
 guard();cmd=[sys.executable,'-X','utf8','scripts/ssh_project.py',action,path]
 if action=='run':cmd+=['--environment','project']
 if report:cmd+=['--report',report]
 result=subprocess.run(cmd,cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=timeout)
 label=str(len(HISTORY))+'_'+action+'_'+Path(path).stem
 (ART/(label+'.stdout.log')).write_text(result.stdout,encoding='utf8');(ART/(label+'.stderr.log')).write_text(result.stderr,encoding='utf8')
 HISTORY.append({'action':action,'path':path,'returncode':result.returncode,'utc':now()})
 assert result.returncode==0,('command_failed',action,path,result.stderr[-1200:])
 return json.loads(result.stdout) if action=='run' else None
def ledger(contact,status,report,pid=None):
 p=REPORTS/'VG_CAPTION_PREPARATION_LAST_CHECK.json';d=json.loads(p.read_text(encoding='utf8'))
 d.update(utc=contact,last_successful_remote_contact_utc=contact,status=status,report=report,direct_human_request=True,controller_pid=pid,current_active_compute='vg_focus_repair_r1_20261030' if pid else None,focus_repair_TEST_accessed=False,focus_repair_policy_adopted=False)
 save(p,d)
def update_program():
 r=subprocess.run([sys.executable,'-X','utf8','scripts/update_caption_program_state.py'],cwd=ROOT,capture_output=True,text=True,encoding='utf8',errors='replace',timeout=60)
 (ART/'program_state_update.log').write_text(r.stdout+r.stderr,encoding='utf8');assert r.returncode==0
try:
 save(ART/'authorization.json',{'utc':now(),'pid':os.getpid(),'authorization':'Human1Nov2026 approved verified V2 cache offload and continuation of server work; exact prior981-shard proposal','scope':'No other assets removed, no other processes signalled, no scientific changes or TEST opening, no parent TRAIN rerun','source_script_sha256':sha(Path(__file__))})
 receipt_path=REPORTS/'VG_V2_CACHE_OFFLOAD_BACKUP_VERIFICATION_20261101.json';state('WAITING_FOR_ALL981_VERIFIED_LOCAL_ARCHIVES')
 while not receipt_path.exists():
  guard();progress=REPORTS/'VG_V2_CACHE_OFFLOAD_PROGRESS_20261101.json'
  if progress.exists():
   data=json.loads(progress.read_text(encoding='utf8'));assert not data.get('status','').startswith('BACKUP_FAILED'),('backup_failed',data)
   age=time.time()-progress.stat().st_mtime;assert age<600,'local_backup_stale_no_deletion'
   state('WAITING_FOR_ALL981_VERIFIED_LOCAL_ARCHIVES',backup_progress=data)
  time.sleep(10)
 rawreceipt=receipt_path.read_text(encoding='utf8');receipt=json.loads(rawreceipt);plan=json.loads((REPORTS/'VG_V2_CACHE_OFFLOAD_PREFLIGHT_20261101.json').read_text(encoding='utf8'))
 assert receipt['status']=='PASS_ALL981_LOCAL_ARCHIVES_RESTORED_TO_ORIGINAL_SHA256' and receipt['verified_shard_count']==981
 assert receipt['cache_manifest_sha256']==plan['cache_manifest_sha256'] and not (REPORTS/'VG_V2_CACHE_OFFLOAD_RESULT_20261101.json').exists()
 state('INDEPENDENT_LOCAL_ARCHIVE_CHECK_BEFORE_SERVER_RETIREMENT')
 expected={e['name']:e for e in plan['registered_shards']};assert set(expected)=={r['name'] for r in receipt['rows']} and len(receipt['rows'])==981
 archivebase=(ROOT/receipt['local_archive_directory']).resolve();assert sha(archivebase/'manifest.json')==plan['cache_manifest_sha256']
 def verify(row):
  path=(ROOT/row['archive_file']).resolve();assert path.is_relative_to(archivebase) and path.name==row['name']+'.bs4.gz'
  assert path.stat().st_size==row['archive_bytes'] and sha(path)==row['archive_sha256']
  with path.open('rb') as f:header=f.read(16)
  assert header[:8]==b'VGBS4GZ1' and struct.unpack('<Q',header[8:16])[0]==row['raw_bytes']==expected[row['name']]['bytes']
  assert row['original_sha256']==expected[row['name']]['expected_sha256'] and row['restored_original_SHA256_pass']
  return True
 with ThreadPoolExecutor(max_workers=4) as pool:assert all(pool.map(verify,receipt['rows']))
 save(ART/'independent_backup_check.json',{'status':'PASS_ALL981_COMPRESSED_FILE_SHA_AND_PRIOR_RAW_ROUNDTRIP','utc':now(),'receipt_sha256':sha(receipt_path),'archive_bytes':receipt['archive_bytes'],'raw_bytes':receipt['raw_bytes']})
 template=(ROOT/'scripts/remote_vg_v2_verified_cache_retirement_20261101.template.py').read_text(encoding='utf8')
 generated=template.replace('__INLINE_APPROVED_PLAN__',repr(plan)).replace('__INLINE_VERIFIED_BACKUP_RECEIPT_RAW__',repr(rawreceipt)).replace('__INLINE_VERIFIED_BACKUP_RECEIPT_SHA__',repr(sha(receipt_path)))
 compile(generated,'verified_retirement','exec');script=ROOT/'scripts/remote_vg_v2_verified_cache_retirement_20261101.py';assert not script.exists();script.write_text(generated,encoding='utf8')
 state('FRESH_CONSUMER_AND_INODE_GUARDS_THEN_EXACT_SERVER_RETIREMENT')
 result=run('run','scripts/remote_vg_v2_verified_cache_retirement_20261101.py','reports/VG_V2_CACHE_OFFLOAD_RESULT_20261101.json',timeout=300)
 assert result['status']=='COMPLETE_981_VERIFIED_V2_SHARDS_OFFLOADED_AND_SERVER_COPIES_REMOVED' and result['deleted_files']==981
 assert result['manifest_preserved_sha256']==receipt['cache_manifest_sha256'] and result['local_backup_receipt_sha256']==sha(receipt_path)
 ledger(result['utc'],'V2_CACHE_OFFLOAD_COMPLETE_VERIFIED','reports/VG_V2_CACHE_OFFLOAD_RESULT_20261101.json')
 save(ART/'verified_retirement_result.json',result);state('POST_CLEANUP_RESOURCE_ADMISSION')
 resource=run('run','scripts/remote_vg_caption_generation_admission_v1.py','reports/VG_FOCUS_REPAIR_R1_RESOURCE_ADMISSION_20261030.json')
 ledger(resource['utc'],'OFFLOAD_COMPLETE_R1_RESOURCE_ADMISSION','reports/VG_FOCUS_REPAIR_R1_RESOURCE_ADMISSION_20261030.json')
 if not resource['admitted'] or resource['aggregate_artifacts_bytes_deduplicated_inodes']+1024**3>40*1024**3:
  update_program();d=state('OFFLOAD_COMPLETE_R1_WAITING_RESOURCE_ADMISSION',resource=resource,cleanup=result);d['status']='COMPLETE_OFFLOAD_R1_NOT_YET_ADMITTED';save(STATUS,d)
 else:
  manifest_path=ROOT/'configs/vg_focus_repair_r1_source_manifest_20261030.json';manifest=json.loads(manifest_path.read_text(encoding='utf8'));assert sha(manifest_path)=='048951e35eee224282fbb885ccdb099e3176a1e0d15a7cb423871f3768406f22'
  for p,h in manifest['source_sha256'].items():assert sha(ROOT/p)==h,('R1_frozen_source_changed',p)
  assert not (REPORTS/'VG_FOCUS_REPAIR_R1_LAUNCH_20261030.json').exists()
  files=['scripts/remote_vg_focus_repair_generate_r1_20261030.py','scripts/remote_vg_focus_repair_score_r1_20261030.py','scripts/remote_vg_focus_repair_controller_r1_20261030.py','scripts/remote_launch_vg_focus_repair_r1_20261030.py','scripts/remote_vg_focus_repair_status_r1_20261030.py','configs/vg_focus_repair_r1_20261030.json','research/VG_FOCUS_REPAIR_FAILURE_AND_R1_PREREGISTRATION_20261030.md','configs/vg_focus_repair_r1_source_manifest_20261030.json']
  for path in files:state('SYNC_FROZEN_R1_AFTER_ADMISSION',sync_path=path);run('sync',path)
  state('EXCLUSIVE_R1_CHECK_ONLY_LAUNCH')
  launch=run('run','scripts/remote_launch_vg_focus_repair_r1_20261030.py','reports/VG_FOCUS_REPAIR_R1_LAUNCH_20261030.json');assert launch['status']=='LAUNCHED'
  snapshot=run('run','scripts/remote_vg_focus_repair_status_r1_20261030.py','reports/VG_FOCUS_REPAIR_R1_LAST_STATUS_20261030.json');active=snapshot.get('status.json',{})
  assert not active.get('TEST_accessed',False)
  verified={'status':'PASS_R1_LAUNCH_AND_INITIAL_STATUS','utc':now(),'launch':launch,'snapshot':snapshot,'TRAIN_reused_not_regenerated':True,'TEST_access_allowed':False}
  save(REPORTS/'VG_FOCUS_REPAIR_R1_INITIAL_VERIFICATION_20261101.json',verified)
  ledger(verified['utc'],'V2_OFFLOAD_COMPLETE_R1_LAUNCH_VERIFIED','reports/VG_FOCUS_REPAIR_R1_INITIAL_VERIFICATION_20261101.json',launch['controller_pid']);update_program()
  d=state('OFFLOAD_COMPLETE_R1_LAUNCHED_VERIFIED',cleanup=result,R1=verified);d['status']='COMPLETE_OFFLOAD_AND_R1_LAUNCH';save(STATUS,d)
 print(json.dumps({k:v for k,v in d.items() if k not in ['history','R1']}),flush=True)
except Exception as e:
 d=state('FAILED_NO_AUTOMATIC_RETRY',error_type=type(e).__name__,reason=str(e)[:1800]);d['status']='FAILED_NO_AUTOMATIC_RETRY';save(STATUS,d);(ART/'failure_traceback.log').write_text(traceback.format_exc(),encoding='utf8');traceback.print_exc();raise
