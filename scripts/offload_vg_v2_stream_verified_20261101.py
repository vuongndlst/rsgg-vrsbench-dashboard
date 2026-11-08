"""Verified local backup over one bounded read-only SSH byte stream, no remote writes."""
from pathlib import Path
from datetime import datetime,timezone
from concurrent.futures import ThreadPoolExecutor,wait,FIRST_COMPLETED
import json,sys,hashlib,gzip,struct,time,shutil,os,re
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research/tools'))
from p0_audit import connect
PHASE=ROOT/'artifacts/offloaded_vg_v2_features_20261101';REPORT=ROOT/'reports/VG_V2_CACHE_OFFLOAD_BACKUP_VERIFICATION_20261101.json';PROGRESS=ROOT/'reports/VG_V2_CACHE_OFFLOAD_PROGRESS_20261101.json'
pre=json.loads((ROOT/'reports/VG_V2_CACHE_OFFLOAD_PREFLIGHT_20261101.json').read_text());expected={e['name']:e for e in pre['registered_shards']};assert len(expected)==981 and not REPORT.exists()
assert hashlib.sha256((PHASE/'manifest.json').read_bytes()).hexdigest()==pre['cache_manifest_sha256']
assert shutil.disk_usage(ROOT).free>sum(e['bytes'] for e in expected.values())+1024**3
start=time.monotonic();rows=[];last_output=0;contact=None;client=None

def now():return datetime.now(timezone.utc).isoformat()
def save(p,value):
 temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2),encoding='utf8');os.replace(temp,p)
def decode(b):
 assert b[:8]==b'VGBS4GZ1';n=struct.unpack('<Q',b[8:16])[0];assert n<32*1024**2;body=gzip.decompress(b[16:]);assert len(body)==n;end=n-n%4
 return np.frombuffer(body[:end],dtype=np.uint8).reshape(4,-1).T.copy().tobytes()+body[end:]
def verify_or_write(name,raw=None):
 e=expected[name];target=PHASE/(name+'.bs4.gz');reused=target.exists()
 if raw is not None:assert len(raw)==e['bytes'] and hashlib.sha256(raw).hexdigest()==e['expected_sha256']
 if not reused:
  assert raw is not None;n=len(raw);end=n-n%4;body=np.frombuffer(raw[:end],dtype=np.uint8).reshape(-1,4).T.copy().tobytes()+raw[end:]
  encoded=b'VGBS4GZ1'+struct.pack('<Q',n)+gzip.compress(body,compresslevel=6,mtime=0);temp=target.with_suffix(target.suffix+'.partial')
  with temp.open('wb') as f:f.write(encoded);f.flush();os.fsync(f.fileno())
  os.replace(temp,target);del body,encoded
 encoded=target.read_bytes();restored=decode(encoded);assert len(restored)==e['bytes'] and hashlib.sha256(restored).hexdigest()==e['expected_sha256']
 if raw is not None:assert restored==raw
 return {'name':name,'split':e['split'],'raw_bytes':e['bytes'],'original_sha256':e['expected_sha256'],'archive_file':target.relative_to(ROOT).as_posix(),'archive_bytes':len(encoded),'archive_sha256':hashlib.sha256(encoded).hexdigest(),'restored_original_SHA256_pass':True,'reused':reused}
def progress(force=False):
 global last_output
 if force or len(rows)%25==0 or time.monotonic()-last_output>30:
  d={'status':'RUNNING_VERIFIED_LOCAL_BACKUP_SSH_STREAM','utc':now(),'completed':len(rows),'total':981,'raw_bytes_verified':sum(r['raw_bytes'] for r in rows),'archive_bytes':sum(r['archive_bytes'] for r in rows),'elapsed_seconds':time.monotonic()-start,'server_deleted_files':0}
  save(PROGRESS,d);print(json.dumps(d),flush=True);last_output=time.monotonic()
try:
 with ThreadPoolExecutor(max_workers=3) as pool:
  existing=[n for n in expected if (PHASE/(n+'.bs4.gz')).exists()]
  fs={pool.submit(verify_or_write,n) for n in existing}
  while fs:
   done,fs=wait(fs,return_when=FIRST_COMPLETED)
   rows.extend(f.result() for f in done);progress()
  remaining=sorted(set(expected)-{r['name'] for r in rows});progress(True)
  payload="from pathlib import Path\nimport os,json,struct,hashlib,sys,signal,resource,time\nos.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(512*1024**2,512*1024**2));signal.alarm(5400)\nroot=(Path.home()/'rsg-former').resolve();cache=root/'artifacts/vg_method_v2/features';out=sys.stdout.buffer\nassert not cache.is_symlink() and cache.resolve()==cache\nraw=(cache/'manifest.json').read_bytes();m=json.loads(raw);expected={e['path']:e for es in m['splits'].values() for e in es}\nNAMES=__NAMES__\nassert hashlib.sha256(raw).hexdigest()==__MANIFEST_SHA__\ndef header(d):\n b=json.dumps(d).encode();out.write(struct.pack('>I',len(b)));out.write(b);out.flush()\nheader({'status':'STREAM_START','manifest_sha256':hashlib.sha256(raw).hexdigest(),'remaining':len(NAMES)})\nfor name in NAMES:\n assert Path(name).name==name and name in expected and name.endswith('.pt') and time.time()<1794670321.2414355\n e=expected[name];p=cache/name;assert p.resolve().parent==cache and not p.is_symlink() and p.stat().st_size==e['bytes']\n header({'name':name,'bytes':e['bytes'],'sha256':e['sha256']});h=hashlib.sha256();count=0\n with p.open('rb') as f:\n  while True:\n   b=f.read(1024*1024)\n   if not b:break\n   out.write(b);h.update(b);count+=len(b)\n out.flush();assert count==e['bytes'] and h.hexdigest()==e['sha256']\nheader({'status':'STREAM_COMPLETE','sent':len(NAMES)})\n"
  payload=payload.replace('__NAMES__',repr(remaining)).replace('__MANIFEST_SHA__',repr(pre['cache_manifest_sha256']))
  client=connect();chan=client.get_transport().open_session(window_size=64*1024**2,max_packet_size=32768,timeout=60);chan.settimeout(120);chan.exec_command('/usr/bin/python3 -B -');chan.sendall(payload.encode());chan.shutdown_write();stream=chan.makefile('rb')
  def read_exact(n):
   out=bytearray()
   while len(out)<n:
    b=stream.read(min(n-len(out),1024*1024))
    if not b:raise RuntimeError('SSH_stream_ended_early')
    out.extend(b)
   return bytes(out)
  def header():
   n=struct.unpack('>I',read_exact(4))[0];assert 0<n<65536;return json.loads(read_exact(n))
  first=header();assert first['status']=='STREAM_START' and first['manifest_sha256']==pre['cache_manifest_sha256']
  pending=set();seen=set()
  while True:
   meta=header()
   if meta.get('status')=='STREAM_COMPLETE':assert meta['sent']==len(remaining);break
   name=meta['name'];assert name in remaining and name not in seen and meta['bytes']==expected[name]['bytes'] and meta['sha256']==expected[name]['expected_sha256'];seen.add(name)
   raw=read_exact(meta['bytes']);contact=now();assert time.monotonic()-start<7200
   pending.add(pool.submit(verify_or_write,name,raw));del raw
   if len(pending)>=4:
    done,pending=wait(pending,return_when=FIRST_COMPLETED);rows.extend(f.result() for f in done);progress()
  while pending:
   done,pending=wait(pending,return_when=FIRST_COMPLETED);rows.extend(f.result() for f in done);progress()
  assert seen==set(remaining)
  code=chan.recv_exit_status()
  if code:raise RuntimeError('remote_read_stream_exit_'+str(code))
 assert len(rows)==981 and {r['name'] for r in rows}==set(expected)
 result={'status':'PASS_ALL981_LOCAL_ARCHIVES_RESTORED_TO_ORIGINAL_SHA256','utc':now(),'last_successful_remote_contact_utc':contact or now(),'cache_manifest_sha256':pre['cache_manifest_sha256'],'source_cache_relative':'artifacts/vg_method_v2/features','local_archive_directory':PHASE.relative_to(ROOT).as_posix(),'verified_shard_count':981,'raw_bytes':sum(r['raw_bytes'] for r in rows),'archive_bytes':sum(r['archive_bytes'] for r in rows),'archive_format':'VGBS4GZ1','rows':sorted(rows,key=lambda r:r['name']),'server_writes':0,'server_deleted_files':0,'CPU_compression_workers':3,'remote_read_CPU':1,'elapsed_seconds':time.monotonic()-start,'precision_changed':False,'TEST_accessed':False,'original_absolute_deadline':1794670321.2414355}
 save(REPORT,result);save(PHASE/'backup_verification.json',result);save(PROGRESS,{k:v for k,v in result.items() if k!='rows'});print(json.dumps({k:v for k,v in result.items() if k!='rows'}),flush=True)
except Exception as e:
 save(PROGRESS,{'status':'BACKUP_FAILED_NO_SERVER_DELETION','utc':now(),'error_type':type(e).__name__,'reason':str(e)[:200]});raise
finally:
 if client:client.close()
