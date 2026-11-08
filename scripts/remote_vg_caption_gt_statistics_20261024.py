"""Read-only streamed train/validation GT analysis. Never analyze TEST phrase semantics."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import gc,hashlib,json,math,os,re,resource,signal,sqlite3,time,zipfile
START=time.monotonic();ROOT=Path.home()/'rsg-former';os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2));signal.alarm(100)
ref=ROOT/'data/vg_caption/region_descriptions.json';h=hashlib.sha256()
with ref.open('rb') as f:
 for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
assert h.hexdigest()=='19624f10afb10bf69e30ecf48d85b1a71420c7432e8c88f1011a3c4fd60ec768'
raw=(ROOT/'artifacts/vg_method_v2/features/manifest.json').read_bytes();assert hashlib.sha256(raw).hexdigest()=='f978799bbd19047b6a74f291aa816f43052251f9b84751f633e8ddf04622abf1'
manifest=json.loads(raw);cache_ids={s:{identity for e in manifest['splits'][s] for identity in e['image_ids']} for s in ['train','val']};split_ids={};split_source_sha={}
with zipfile.ZipFile(ROOT/'data/vg_canonical_audit/reltr_vg_annotations.zip') as z:
 for split in ['train','val']:
  raw=z.read('vg/'+split+'.json');split_source_sha[split]=hashlib.sha256(raw).hexdigest();meta=json.loads(raw);split_ids[split]={int(x['id']) for x in meta['images']};del raw,meta;gc.collect()
assert len(split_ids['train'])==57723 and len(split_ids['val'])==5000 and not split_ids['train']&split_ids['val']
cache_membership_diagnostic={s:{'cached_ids':len(cache_ids[s]),'canonical_ids':len(split_ids[s]),'canonical_missing_from_cache':sorted(split_ids[s]-cache_ids[s]),'cache_outside_canonical':sorted(cache_ids[s]-split_ids[s])} for s in split_ids}
db=sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON');dims={i:(w,h) for i,w,h in db.execute('SELECT vg_id,width,height FROM pixels')};db.close()
colors=set('black white red green blue yellow brown gray grey orange pink purple silver gold golden tan beige blond blonde'.split());materials=set('wooden wood metal metallic plastic glass brick concrete leather denim cotton'.split());spatial=set('on in under above behind beside near between inside outside next along front at'.split())
def fresh():return {'images':0,'raw_regions':0,'eligible_regions':0,'rejected':Counter(),'region_counts':Counter(),'lengths':Counter(),'area_bins':Counter(),'terms':Counter(),'phrases_with_color':0,'phrases_with_material':0,'phrases_with_spatial_cue':0,'phrases_with_final_period':0,'phrases_containing_and':0,'duplicate_normalized_phrases_within_image':0,'duplicate_geometry_different_region_ids':0,'duplicate_region_ids':0,'parent_identity_mismatches':0,'empty_phrases':0,'missing_dimensions_images':0,'examples':[]}
stats={s:fresh() for s in split_ids};sample_ids=sorted(split_ids['val'],key=lambda x:hashlib.sha256(('GTSTYLE20261024:'+str(x)).encode()).hexdigest())[:12];sample_ids=set(sample_ids)
decoder=json.JSONDecoder();buf='';scanned=0;bytes_read=0;complete=False;seen_images=set()
with ref.open(encoding='utf8') as f:
 assert f.read(1)=='['
 while time.monotonic()-START<75:
  buf=buf.lstrip()
  if buf.startswith(','):buf=buf[1:].lstrip()
  if buf.startswith(']'):complete=True;break
  try:item,end=decoder.raw_decode(buf)
  except json.JSONDecodeError:
   part=f.read(1024**2);assert part,'unexpected_eof';bytes_read+=len(part.encode('utf8'));buf+=part;assert len(buf)<8*1024**2;continue
  buf=buf[end:];scanned+=1;identity=item['id'];assert identity not in seen_images;seen_images.add(identity)
  split='train' if identity in split_ids['train'] else 'val' if identity in split_ids['val'] else None
  if split is None:continue
  st=stats[split];rows=item['regions'];st['images']+=1;st['raw_regions']+=len(rows);st['region_counts'][len(rows)]+=1;seen=set();geometries=set();phrases=set()
  if identity not in dims:st['missing_dimensions_images']+=1
  W,H=dims.get(identity,(None,None))
  for row in rows:
   rid=row['region_id'];st['duplicate_region_ids']+=rid in seen;seen.add(rid);st['parent_identity_mismatches']+=row['image_id']!=identity
   phrase=row.get('phrase');valid_phrase=isinstance(phrase,str) and bool(phrase.strip());st['empty_phrases']+=not valid_phrase
   vals=[row[k] for k in ['x','y','width','height']];reason=None
   if not valid_phrase:reason='empty_phrase'
   elif identity not in dims:reason='missing_audited_dimensions'
   elif any(type(x) not in [int,float] or not math.isfinite(x) for x in vals):reason='invalid_numeric_geometry'
   elif vals[0]<0 or vals[1]<0 or vals[2]<=0 or vals[3]<=0:reason='negative_or_nonpositive_geometry'
   elif vals[0]+vals[2]>W or vals[1]+vals[3]>H:reason='box_outside_image'
   if reason:st['rejected'][reason]+=1;continue
   st['eligible_regions']+=1;geometry=tuple(vals);st['duplicate_geometry_different_region_ids']+=geometry in geometries;geometries.add(geometry)
   tokens=re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+",phrase.lower());normalized=' '.join(tokens);st['duplicate_normalized_phrases_within_image']+=normalized in phrases;phrases.add(normalized);st['lengths'][len(tokens)]+=1;st['terms'].update(tokens);ts=set(tokens)
   st['phrases_with_color']+=bool(ts&colors);st['phrases_with_material']+=bool(ts&materials);st['phrases_with_spatial_cue']+=bool(ts&spatial);st['phrases_with_final_period']+=phrase.rstrip().endswith('.');st['phrases_containing_and']+='and' in ts
   area=vals[2]*vals[3]/(W*H);st['area_bins']['<=0.01' if area<=.01 else '<=0.05' if area<=.05 else '<=0.25' if area<=.25 else '<=0.5' if area<=.5 else '>0.5']+=1
   if split=='val' and identity in sample_ids and len(st['examples'])<36 and len([e for e in st['examples'] if e['image_id']==identity])<3:st['examples'].append({'image_id':identity,'region_id':rid,'box_xywh':vals,'phrase':phrase,'token_count':len(tokens)})
def quantile(hist,q):
 threshold=q*sum(hist.values());n=0
 for value,count in sorted(hist.items()):
  n+=count
  if n>=threshold:return value
 return None
out={}
for split,st in stats.items():
 n=st['eligible_regions'];st['expected_images']=len(split_ids[split]);st['image_membership_complete']=st['images']==len(split_ids[split]);st['length_summary']={'mean':sum(k*v for k,v in st['lengths'].items())/max(n,1),'p10':quantile(st['lengths'],.1),'median':quantile(st['lengths'],.5),'p90':quantile(st['lengths'],.9),'p95':quantile(st['lengths'],.95),'one_to_three_words':sum(v for k,v in st['lengths'].items() if k<=3),'over_20_words':sum(v for k,v in st['lengths'].items() if k>20)};st['regions_per_image_summary']={'mean':st['raw_regions']/max(st['images'],1),'median':quantile(st['region_counts'],.5),'p90':quantile(st['region_counts'],.9)};st['top_terms']=st.pop('terms').most_common(80);st['length_histogram']=dict(sorted(st.pop('lengths').items()));st.pop('region_counts');out[split]=st
 assert st['eligible_regions']+sum(st['rejected'].values())==st['raw_regions']
print(json.dumps({'status':'TRAIN_VAL_GT_STATISTICS_COMPLETE' if complete else 'TRAIN_VAL_GT_STATISTICS_PARTIAL_TIME_CAP','utc':datetime.now(timezone.utc).isoformat(),'source_sha256':h.hexdigest(),'split_annotation_sha256':split_source_sha,'cache_membership_diagnostic':cache_membership_diagnostic,'read_only':True,'server_files_written':0,'test_reference_semantics_analyzed':False,'test_captions_generated':False,'test_metrics_scored':False,'source_image_records_scanned':scanned,'source_stream_complete':complete,'source_stream_bytes_read':bytes_read,'splits':out,'phrase_length_tokenization':'ASCII word/apostrophe-or-number regex; descriptive audit, not metric PTB tokenizer','lexical_cues':'explicit limited dictionaries; not a POS/attribute ground-truth annotation or model coverage measurement','wall_seconds':time.monotonic()-START,'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}))
