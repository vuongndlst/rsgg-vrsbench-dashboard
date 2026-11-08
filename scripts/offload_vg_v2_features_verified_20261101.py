"""Authorized local FP32 archive: bounded read-only SFTP, no server removals."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,as_completed
import json,sys,hashlib,gzip,struct,time,threading,shutil,os
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research/tools'))
from p0_audit import connect
PHASE=ROOT/'artifacts/offloaded_vg_v2_features_20261101';REPORT=ROOT/'reports/VG_V2_CACHE_OFFLOAD_BACKUP_VERIFICATION_20261101.json';PROGRESS=ROOT/'reports/VG_V2_CACHE_OFFLOAD_PROGRESS_20261101.json'
pre=json.loads((ROOT/'reports/VG_V2_CACHE_OFFLOAD_PREFLIGHT_20261101.json').read_text());expected=pre['registered_shards'];assert len(expected)==981 and not pre['unexpected_entries']
assert all(Path(e['name']).name==e['name'] and e['name'].endswith('.pt') and e['bytes']<32*1024**2 for e in expected)
assert not REPORT.exists(),'backup_already_complete_use_existing_verification'
assert shutil.disk_usage(ROOT).free>sum(e['bytes'] for e in expected)+2*1024**3
start=time.monotonic();stop=threading.Event();tls=threading.local();clients=[];mutex=threading.Lock();contacts=[]
def now():return datetime.now(timezone.utc).isoformat()
def save(path,value):
 temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(temp,path)
def get_session():
 if not hasattr(tls,'sftp'):
  client=connect();s=client.open_sftp();s.get_channel().settimeout(60);home=s.normalize('.')
  tls.sftp=s;tls.base=home+'/rsg-former/artifacts/vg_method_v2/features/'
  with mutex:clients.append(client)
 return tls.sftp,tls.base
def restore(encoded):
 assert encoded[:8]==b'VGBS4GZ1';n=struct.unpack('<Q',encoded[8:16])[0];assert n<=32*1024**2
 body=gzip.decompress(encoded[16:]);assert len(body)==n;end=n-n%4
 return np.frombuffer(body[:end],dtype=np.uint8).reshape(4,-1).T.copy().tobytes()+body[end:]
try:
 s,base=get_session()
 with s.open(base+'manifest.json','rb') as f:rawmanifest=f.read()
 assert hashlib.sha256(rawmanifest).hexdigest()==pre['cache_manifest_sha256'];manifest=json.loads(rawmanifest)
 current={e['path']:e for es in manifest['splits'].values() for e in es}
 assert set(current)=={e['name'] for e in expected}
 for e in expected:assert current[e['name']]['sha256']==e['expected_sha256'] and current[e['name']]['bytes']==e['bytes']
 PHASE.mkdir(parents=True,exist_ok=True)
 mp=PHASE/'manifest.json'
 if mp.exists():assert mp.read_bytes()==rawmanifest
 else:mp.write_bytes(rawmanifest)
 save(PHASE/'authorization_and_scope.json',{'utc':now(),'authorization':'Human1Nov2026 directly approved saving V2 cache locally and continuing server work after proposed981-shard backup/SHA verification/removal','scope':'981manifest-listed .pt shards only; manifest/checkpoints/source/config/logs/predictions preserved','archive_format':'VGBS4GZ1 + uint64LE raw length + gzip6 four byte planes and tail','source_manifest_sha256':pre['cache_manifest_sha256'],'server_writes_or_deletions_by_this_script':False,'workers':2,'SFTP_prefetch_requests_per_worker':32})
 def one(e):
  if stop.is_set():raise RuntimeError('prior_transfer_failed')
  assert time.monotonic()-start<7200,'bounded_transfer_wall'
  assert time.time()<1794670321.2414355,'original_absolute_deadline'
  target=PHASE/(e['name']+'.bs4.gz');reused=False
  if target.exists():
   encoded=target.read_bytes();raw=restore(encoded);reused=True
  else:
   sftp,remote=get_session();st=sftp.stat(remote+e['name']);assert st.st_size==e['bytes']
   with sftp.open(remote+e['name'],'rb') as f:
    f.prefetch(e['bytes'],max_concurrent_requests=32);raw=f.read(e['bytes']+1)
   with mutex:contacts.append(now())
   assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['expected_sha256'],e['name']
   n=len(raw);end=n-n%4;shuffled=np.frombuffer(raw[:end],dtype=np.uint8).reshape(-1,4).T.copy().tobytes()+raw[end:]
   encoded=b'VGBS4GZ1'+struct.pack('<Q',n)+gzip.compress(shuffled,compresslevel=6,mtime=0);del shuffled
   temp=target.with_suffix(target.suffix+'.partial')
   with temp.open('wb') as dst:dst.write(encoded);dst.flush();os.fsync(dst.fileno())
   os.replace(temp,target);del encoded
   encoded=target.read_bytes();restored=restore(encoded)
   assert restored==raw and hashlib.sha256(restored).hexdigest()==e['expected_sha256'];del restored
  assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['expected_sha256']
  return {'name':e['name'],'split':e['split'],'raw_bytes':len(raw),'original_sha256':e['expected_sha256'],'archive_file':target.relative_to(ROOT).as_posix(),'archive_bytes':len(encoded),'archive_sha256':hashlib.sha256(encoded).hexdigest(),'restored_original_SHA256_pass':True,'reused':reused}
 rows=[];last=0
 save(PROGRESS,{'status':'RUNNING_VERIFIED_LOCAL_BACKUP','utc':now(),'completed':0,'total':981,'server_deleted_files':0})
 with ThreadPoolExecutor(max_workers=2) as pool:
  futures={pool.submit(one,e):e['name'] for e in expected}
  for f in as_completed(futures):
   try:rows.append(f.result())
   except Exception:stop.set();raise
   if len(rows)%25==0 or time.monotonic()-last>=30:
    summary={'status':'RUNNING_VERIFIED_LOCAL_BACKUP','utc':now(),'completed':len(rows),'total':981,'raw_bytes_verified':sum(e['raw_bytes'] for e in rows),'archive_bytes':sum(e['archive_bytes'] for e in rows),'elapsed_seconds':time.monotonic()-start,'server_deleted_files':0}
    save(PROGRESS,summary);print(json.dumps(summary),flush=True);last=time.monotonic()
 assert len(rows)==981 and {r['name'] for r in rows}==set(current)
 # Every successful worker verified the bytes read back from its durable local archive.
 result={'status':'PASS_ALL981_LOCAL_ARCHIVES_RESTORED_TO_ORIGINAL_SHA256','utc':now(),'last_successful_remote_contact_utc':max(contacts) if contacts else now(),'cache_manifest_sha256':hashlib.sha256(rawmanifest).hexdigest(),'source_cache_relative':'artifacts/vg_method_v2/features','local_archive_directory':PHASE.relative_to(ROOT).as_posix(),'verified_shard_count':981,'raw_bytes':sum(e['raw_bytes'] for e in rows),'archive_bytes':sum(e['archive_bytes'] for e in rows),'archive_format':'VGBS4GZ1','rows':sorted(rows,key=lambda r:r['name']),'server_writes':0,'server_deleted_files':0,'CPU_workers':2,'elapsed_seconds':time.monotonic()-start,'precision_changed':False,'TEST_accessed':False,'original_absolute_deadline':1794670321.2414355}
 save(REPORT,result);save(PHASE/'backup_verification.json',result);save(PROGRESS,{k:v for k,v in result.items() if k!='rows'});print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
except Exception as e:
 stop.set();save(PROGRESS,{'status':'BACKUP_FAILED_NO_SERVER_DELETION','utc':now(),'error_type':type(e).__name__,'reason':str(e)[:250],'elapsed_seconds':time.monotonic()-start});raise
finally:
 for c in clients:c.close()
