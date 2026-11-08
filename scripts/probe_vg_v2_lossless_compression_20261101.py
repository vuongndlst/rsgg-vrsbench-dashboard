"""Two fixed TRAIN/VAL cache shards, read-only SFTP and local lossless benchmark."""
from pathlib import Path
from datetime import datetime,timezone
import sys,json,hashlib,time,shutil,gzip,lzma,zipfile,collections
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'research/tools'))
from p0_audit import connect
pre=json.loads((ROOT/'reports/VG_V2_CACHE_OFFLOAD_PREFLIGHT_20261101.json').read_text());selected=[next(x for x in pre['registered_shards'] if x['name']==name) for name in ['train_00000.pt','val_00000.pt']]
out=ROOT/'artifacts/vg_v2_lossless_probe_20261101';assert not out.exists();out.mkdir();assert shutil.disk_usage(out).free>1024**3
c=None
try:
 c=connect();s=c.open_sftp();home=s.normalize('.');base=home+'/rsg-former/artifacts/vg_method_v2/features/'
 with s.open(base+'manifest.json','rb') as f:manifest=f.read()
 assert hashlib.sha256(manifest).hexdigest()==pre['cache_manifest_sha256'];(out/'manifest.json').write_bytes(manifest)
 for row in selected:
  target=out/row['name'];h=hashlib.sha256();count=0
  with s.open(base+row['name'],'rb') as src,target.open('xb') as dst:
   while True:
    b=src.read(1024*1024)
    if not b:break
    count+=len(b);assert count<=32*1024**2;h.update(b);dst.write(b)
  assert count==row['bytes'] and h.hexdigest()==row['expected_sha256']
 contact=datetime.now(timezone.utc).isoformat()
finally:
 if c:c.close()
import torch
torch.set_num_threads(1);torch.set_num_interop_threads(1)
results=[]
for row in selected:
 p=out/row['name'];chunk=torch.load(p,map_location='cpu',weights_only=True);first=chunk[0];tensor_info={k:{'shape':list(v.shape),'dtype':str(v.dtype),'bytes':v.numel()*v.element_size()} for k,v in first.items() if isinstance(v,torch.Tensor)}
 tensors=sum(v.numel()*v.element_size() for item in chunk for v in item.values() if isinstance(v,torch.Tensor))
 raw=p.read_bytes();entry={'split':row['split'],'shard':row['name'],'original_sha256':row['expected_sha256'],'raw_bytes':len(raw),'images':len(chunk),'tensor_layout':tensor_info,'all_tensor_bytes':tensors,'serialization_overhead_bytes':len(raw)-tensors,'algorithms':{}}
 with zipfile.ZipFile(p) as z:entry['torch_zip_compression_methods']=dict(collections.Counter(i.compress_type for i in z.infolist()))
 for name,compress,decompress,extension in [('gzip6',lambda b:gzip.compress(b,compresslevel=6,mtime=0),gzip.decompress,'.gz'),('xz3',lambda b:lzma.compress(b,preset=3),lzma.decompress,'.xz')]:
  start=time.perf_counter();encoded=compress(raw);comp=time.perf_counter()-start;target=out/(row['name']+extension);target.write_bytes(encoded)
  start=time.perf_counter();restored=decompress(encoded);decomp=time.perf_counter()-start;assert hashlib.sha256(restored).hexdigest()==row['expected_sha256'] and restored==raw
  entry['algorithms'][name]={'compressed_bytes':len(encoded),'retained_fraction':len(encoded)/len(raw),'saved_percent':100*(1-len(encoded)/len(raw)),'compression_seconds':comp,'decompression_seconds':decomp,'restored_SHA256_matches_original':True}
  del encoded,restored
 results.append(entry);del raw,chunk
m=json.loads(manifest);counts={split:sum(e['images'] for e in entries) for split,entries in m['splits'].items()}
report={'status':'PASS_TWO_FIXED_SHARD_LOCAL_LOSSLESS_BYTE_PARITY','utc':datetime.now(timezone.utc).isoformat(),'last_successful_remote_contact_utc':contact,'server_writes':0,'files_deleted':0,'extractor_or_model_run':False,'SGDet_or_caption_scoring':False,'TEST_accessed':False,'precision_changed':False,'cache_dtype':m['cache_dtype'],'image_counts':counts,'full_cache_logical_bytes':sum(x['bytes'] for x in pre['registered_shards']),'shards':results,'projection_is_sample_based_not_full_cache_measurement':True}
for alg in ['gzip6','xz3']:
 sample_ratio=sum(x['algorithms'][alg]['compressed_bytes'] for x in results)/sum(x['raw_bytes'] for x in results)
 report[alg+'_sample_projection_GiB']=report['full_cache_logical_bytes']*sample_ratio/1024**3
(ROOT/'reports/VG_V2_LOSSLESS_COMPRESSION_PROBE_20261101.json').write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report))
