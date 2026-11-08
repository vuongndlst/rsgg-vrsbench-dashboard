"""Fixed20 already-observed DEV review artifact; original JPGs only, no inference."""
from pathlib import Path
from datetime import datetime,timezone
import csv,hashlib,io,json,os,shutil,struct,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];CONFIG=ROOT/'configs/vg_grounding_dev_review20_v1_20261104.json';PREREG=ROOT/'reports/VG_GROUNDING_DEV_REVIEW20_PREREGISTRATION_20261104.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 cfg=json.loads(CONFIG.read_text());assert datetime.now(timezone.utc).timestamp()<cfg['overall_deadline'];pr=json.loads(PREREG.read_text());assert sha(Path(__file__))==pr['source_sha256'] and sha(CONFIG)==pr['config_sha256']
 for rel,h in pr['input_sha256'].items():assert sha(ROOT/rel)==h,rel
 assert shutil.disk_usage(ROOT).free>=16*1024**3
 packet=json.loads((ROOT/cfg['packet']).read_text());queries={u['unit_id']:u for u in json.loads((ROOT/cfg['queries']).read_text())}
 assert len(packet)==20 and {p['unit_id'] for p in packet}==set(cfg['fixed_unit_ids'])
 meta={str(i):cfg['images'][str(i)] for i in cfg['unique_parent_images']}
 remote=r"""from pathlib import Path
from datetime import datetime,timezone
import hashlib,io,json,os,resource,signal,sqlite3,struct,sys,zipfile,time
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))});resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2));signal.alarm(75);start=time.time();assert start<1794670321.2414355
wanted=EXPECTED
root=Path.home()/'rsg-former';dbpath=root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024**2),b''):h.update(b)
 return h.hexdigest()
dbh=sha(dbpath);assert dbh=='983789d6425c628ac22849090f6160f8b3e55910da4f1aecad8a5b61a288e6f9'
db=sqlite3.connect(dbpath.as_uri()+'?mode=ro',uri=True);db.execute('PRAGMA query_only=ON')
for identity,m in wanted.items():
 row=db.execute('SELECT split,jpeg_sha,width,height FROM pixels WHERE vg_id=?',(int(identity),)).fetchone()
 assert row==('val',m['jpeg_sha256'],m['width'],m['height'])
db.close();found={};raw_total=0;receipts=[];out=io.BytesIO()
with zipfile.ZipFile(out,'w',zipfile.ZIP_STORED) as target:
 for archive in ['images.zip','images2.zip']:
  path=Path.home()/'sbir/hienlth/vg'/archive;st=path.stat()
  with zipfile.ZipFile(path) as z:
   for entry in z.infolist():
    identity=Path(entry.filename).stem
    if not entry.filename.lower().endswith('.jpg') or identity not in wanted:continue
    assert identity not in found and 0<entry.file_size<8*1024**2
    raw=z.read(entry);actual=hashlib.sha256(raw).hexdigest();assert actual==wanted[identity]['jpeg_sha256']
    target.writestr('images/'+identity+'.jpg',raw);raw_total+=len(raw);assert raw_total<64*1024**2
    found[identity]=actual;receipts.append({'image_id':int(identity),'jpeg_sha256':actual,'source_archive':archive,'source_member':entry.filename,'bytes':len(raw),'width':wanted[identity]['width'],'height':wanted[identity]['height']})
  assert path.stat().st_size==st.st_size and path.stat().st_mtime_ns==st.st_mtime_ns
 assert set(found)==set(wanted) and sha(dbpath)==dbh and time.time()-start<75
 manifest={'status':'FIXED_DEV_SOURCE_JPEGS_READ_ONLY_SHA_VERIFIED','utc':datetime.now(timezone.utc).isoformat(),'image_count':len(found),'images':receipts,'source_pixel_database_sha256':dbh,'raw_JPEG_bytes':raw_total,'no_model_or_metric':True,'server_files_modified':False,'TEST_images_or_references_read':False,'original_archive_members_read_only':True,'wall_seconds':time.time()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
 target.writestr('source_manifest.json',json.dumps(manifest,indent=2))
raw=out.getvalue();assert len(raw)<65*1024**2;sys.stdout.buffer.write(b'VGRVW001'+struct.pack('<Q',len(raw))+raw);sys.stdout.buffer.flush()
""".replace('EXPECTED',repr(meta))
 art=ROOT/cfg['archive_directory'];out=ROOT/cfg['output_directory'];assert not art.exists() and not out.exists();art.mkdir(parents=True);out.mkdir(parents=True)
 sys.path.insert(0,str(ROOT/'research/tools'));from p0_audit import connect
 client=connect()
 try:
  stdin,stdout,stderr=client.exec_command('PYTHONDONTWRITEBYTECODE=1 /usr/bin/python3 -B -',timeout=100);stdin.write(remote);stdin.channel.shutdown_write()
  header=stdout.read(16);assert header[:8]==b'VGRVW001','remote_readonly_image_stream_failed'
  n=struct.unpack('<Q',header[8:])[0];assert n<65*1024**2
  raw=stdout.read(n);assert len(raw)==n;assert stdout.channel.recv_exit_status()==0,stderr.read().decode(errors='replace')[-300:]
  (art/'source_images.zip').write_bytes(raw)
 finally:client.close()
 fetched=datetime.now(timezone.utc).isoformat()
 from PIL import Image
 with zipfile.ZipFile(io.BytesIO(raw)) as z:
  sm=json.loads(z.read('source_manifest.json'));assert sm['image_count']==len(meta)
  for r in sm['images']:
   name='images/'+str(r['image_id'])+'.jpg';data=z.read(name);assert hashlib.sha256(data).hexdigest()==r['jpeg_sha256']
   with Image.open(io.BytesIO(data)) as im:assert im.size==(r['width'],r['height'])
   dest=out/name;dest.parent.mkdir(exist_ok=True);dest.write_bytes(data)
 (out/'source_manifest.json').write_text(json.dumps(sm,indent=2),encoding='utf8')
 units=[]
 for row in packet:
  q=queries[row['unit_id']];assert row['query_graph']==q['graph'] and q['split']=='val' and q['parent_image_id'] in cfg['unique_parent_images']
  units.append({**row,'parent_image_id':q['parent_image_id'],'image_size':q['image_size'],'crop_xyxy':q['crop_xyxy']})
 payload=json.dumps(units,ensure_ascii=True).replace('<',chr(92)+'u003c')
 template=(ROOT/'scripts/vg_grounding_review20_template_v1_20261104.html').read_text(encoding='utf8')
 html=template.replace('/*__REVIEW_DATA__*/',payload);assert '/*__REVIEW_DATA__*/' not in html
 (out/'index.html').write_text(html,encoding='utf8')
 (out/'review_data.json').write_text(json.dumps({'status':'UNBLINDED_DEV_REVIEW_PACKET_NOT_HUMAN_LABELS','units':units,'human_annotations_created':0,'annotation_policy':'blank until reviewer enters findings; no automatic accuracy gate'},indent=2),encoding='utf8')
 with (out/'blank_annotation.csv').open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.writer(f);w.writerow(['unit_id','parent_image_id','reviewer','reviewed','visible_target_match','relation_supported','extra_detail','missing_target_or_facts','uncertain','note'])
  for r in units:w.writerow([r['unit_id'],r['parent_image_id'],'','','','','','','',''])
 result={'status':'FIXED20_DEV_VISUAL_REVIEW_PACKET_COMPLETE_NOT_HUMAN_EVALUATION','utc':datetime.now(timezone.utc).isoformat(),'phase':cfg['phase'],'unit_count':20,'source_images':len(meta),'source_JPEG_bytes':sm['raw_JPEG_bytes'],'original_JPEG_hashes_verified':True,'image_dimensions_verified':True,'pre_fixed_sample_not_cherry_picked':True,'all_three_arms_and_original_references_present':True,'no_new_inference_scoring_or_selection':True,'human_labels_created':0,'visual_semantic_labels_created':0,'review_is_unblinded':True,'images_are_original_not_generated_or_modified':True,'check_or_TEST_opened':False,'source_images_zip_sha256':sha(art/'source_images.zip'),'preregistration_sha256':sha(PREREG),'local_output_directory':cfg['output_directory'],'successful_remote_image_contact_utc':sm['utc'],'successful_binary_fetch_acknowledged_utc':fetched,'remote_readonly_wall_seconds':sm['wall_seconds'],'output_sha256':{p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file()}}
 assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<70*1024**2
 (ROOT/'reports/VG_GROUNDING_DEV_REVIEW20_RESULT_20261104.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k!='output_sha256'}))
if __name__=='__main__':main()
