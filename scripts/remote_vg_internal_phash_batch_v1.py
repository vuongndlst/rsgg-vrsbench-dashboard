"""Bounded resumable internal VG candidate audit; original journals read-only."""
from pathlib import Path
from datetime import datetime,timezone
from contextlib import closing
import hashlib,json,os,resource,shutil,signal,sqlite3,time

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

class BatchDeadline(Exception):pass

def check_deadline(deadline):
 if deadline is not None and time.monotonic()>=deadline:raise BatchDeadline()

def candidate_ids(db, hashes, deadline=None):
 from rsgformer.caption_phash_candidates_v1 import bands
 keys={key for encoded in hashes for key in bands(encoded)}
 ids=set()
 for band,number in sorted(keys):
  check_deadline(deadline)
  ids.update(row[0] for row in db.execute('SELECT vg_id FROM bands WHERE band=? AND value=?',(band,number)))
 return sorted(ids)

def build_band_index(db,rows,deadline):
 from rsgformer.caption_phash_candidates_v1 import bands
 last=db.execute('SELECT value FROM state WHERE key="last_band_id"').fetchone()
 last_id=int(last[0]) if last else -1
 processed=0
 for identity,split,*hs in rows:
  if split not in ('val','test') or identity<=last_id:continue
  if time.monotonic()>=deadline:break
  for rotation,encoded in enumerate(hs):
   db.executemany('INSERT INTO bands VALUES (?,?,?,?)',[(band,number,identity,rotation*90) for band,number in bands(encoded)])
  db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',('last_band_id',str(identity)))
  processed+=1
  if processed%128==0:db.commit()
 else:db.execute('INSERT INTO state VALUES (?,?)',('band_index_complete','true'))
 db.commit()

def process_queries(db,rows,retained,deadline,limit=1024,pair_cap=200000):
 from rsgformer.caption_phash_candidates_v1 import rotation_match,protected_direction
 query_rows=[r for r in rows if r[1] in ('train','val')];by_id={r[0]:r for r in rows}
 last=db.execute('SELECT value FROM state WHERE key="last_query_id"').fetchone()
 last_id=int(last[0]) if last else -1
 processed=0;pair_count=db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]
 for identity,split,*hs in query_rows:
  if identity<=last_id:continue
  if processed>=limit or time.monotonic()>=deadline:break
  added=0
  try:
   for target_id in candidate_ids(db,hs,deadline):
    check_deadline(deadline)
    target=by_id[target_id]
    if not protected_direction(split,'vg',target[1]):continue
    match=rotation_match(hs,target[2:])
    if match:
     assert pair_count+added< pair_cap,'pair_cap_no_identity_omission'
     db.execute('INSERT INTO pairs VALUES (?,?,?,?,?,?)',(identity,target_id,*match,int(identity in retained[split])))
     added+=1
   db.execute('INSERT OR REPLACE INTO state VALUES (?,?)',('last_query_id',str(identity)))
   db.commit()
  except BatchDeadline:
   db.rollback();break
  except Exception:
   db.rollback();raise
  last_id=identity;processed+=1;pair_count+=added
 return processed,last_id,query_rows

def snapshot(path):
 with closing(sqlite3.connect(path.as_uri()+'?mode=ro',uri=True,timeout=5)) as db:
  db.execute('PRAGMA query_only=ON');db.execute('BEGIN')
  source=json.loads(db.execute('SELECT value FROM provenance WHERE key="source"').fetchone()[0])
  rows=db.execute('SELECT vg_id,split,phash0,phash90,phash180,phash270 FROM pixels ORDER BY vg_id').fetchall()
 return source,rows

def main():
 started=time.monotonic()
 def alarm_timeout(signum,frame):raise TimeoutError('batch_alarm_100_seconds')
 signal.signal(signal.SIGALRM,alarm_timeout);signal.alarm(100)
 os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
 resource.setrlimit(resource.RLIMIT_AS,(768*1024**2,768*1024**2))
 root=Path.home()/'rsg-former';base=root/'artifacts/caption_readiness'
 folder=base/'vg_internal_phash_candidates_v1';journal=folder/'candidates.sqlite'
 from rsgformer.caption_phash_candidates_v1 import bands,value,rotation_match,protected_direction
 assert shutil.disk_usage(root).free>16*1024**3+128*1024**2
 names=['scripts/remote_vg_internal_phash_batch_v1.py','src/rsgformer/caption_phash_candidates_v1.py','configs/caption_perceptual_exclusion_v1.json']
 hashes={name:sha(root/name) for name in names}
 assert hashes[names[1]]=='827ac8e5a8ce4b0c8921b606b7feadf33c9df5f721f39f2a5f659f8b532f6688'
 assert hashes[names[2]]=='f3efd67929926cbad2c76715d0ae768fa750dead3af41c81033d4951f0e9195b'
 assert sha(root/'scripts/remote_vg_pixel_fingerprint_batch.py')=='5573148955570b19bc9f8e3c6834462c4ac9e2709f299e5b1ba3cc74ea48622c'
 assert json.loads((root/'reports/VG_PIXEL_FINGERPRINT_PROGRESS.json').read_text())['status']=='VG_PIXEL_INDEX_COMPLETE'
 source_path=base/'vg_pixel_fingerprints_v1.sqlite';before_sha=sha(source_path)
 source,rows=snapshot(source_path)
 assert source['script_sha256']=='5573148955570b19bc9f8e3c6834462c4ac9e2709f299e5b1ba3cc74ea48622c'
 assert source['canonical_annotation_sha256']=='56856b4386220f40d6b1b522a2ae199a8b4188765fd9ebb5321c1d0f4f2d7fbc'
 assert source['method']=='raw Pillow RGB/no EXIF; SHA256(width,height,RGB); normalized DCT32 low8 median excluding DC phash63; 0/90/180/270 rotations'
 assert len(rows)==89169 and len({r[0] for r in rows})==89169
 counts={s:sum(r[1]==s for r in rows) for s in ['train','val','test']}
 assert counts=={'train':57723,'val':5000,'test':26446}
 for row in rows:
  for encoded in row[2:]:value(encoded)
 assert sha(source_path)==before_sha,'original_index_changed_during_snapshot'
 draft_path=base/'canonical_internal_external_identity_splits_v2.json'
 assert sha(draft_path)=='ade090fd4975e12e1ade2ac16a734f7b167f6630b38d1710cefa4176523f7b16'
 draft=json.loads(draft_path.read_text())['split_image_ids'];retained={s:set(ids) for s,ids in draft.items()}
 protocol={'protocol_id':'vg_internal_phash_candidates_v1','source_hashes':hashes,
           'vg_source_database_sha256':before_sha,'vg_provenance':source,
           'vg_rows_sha256':hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),
           'draft_manifest_sha256':sha(draft_path),'radius':4,'rotation_pairs':16,
           'query_splits':['train','val'],'protected_splits':['val','test'],
           'cpu_threads':1,'address_space_bytes_cap':768*1024**2,'journal_bytes_cap':128*1024**2,
           'queries_per_batch_cap':1024,'soft_wall_seconds':75,'alarm_seconds':100,
           'matching_pairs_cap':200000,'external_coverage_in_this_phase':False,
           'original_journal_modified':False,'training_manifest_modified':False,'test_scored':False}
 raw=json.dumps(protocol,sort_keys=True,separators=(',',':'))
 folder.mkdir(exist_ok=True)
 lock=folder/'preregistration.json'
 if lock.exists():assert lock.read_text()==raw,'resume_provenance_changed_new_phase_required'
 else:lock.write_text(raw)
 with closing(sqlite3.connect(journal,timeout=5)) as db:
  db.execute('PRAGMA page_size=4096');db.execute('PRAGMA max_page_count=32768');db.execute('PRAGMA cache_size=-8192')
  db.executescript('CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY,value TEXT NOT NULL); CREATE TABLE IF NOT EXISTS bands(band INTEGER,value INTEGER,vg_id INTEGER,rotation INTEGER,PRIMARY KEY(band,value,vg_id,rotation)) WITHOUT ROWID; CREATE TABLE IF NOT EXISTS pairs(source_id INTEGER,target_id INTEGER,distance INTEGER,source_rotation INTEGER,target_rotation INTEGER,retained_draft INTEGER,PRIMARY KEY(source_id,target_id)) WITHOUT ROWID;')
  stored=db.execute('SELECT value FROM state WHERE key="protocol"').fetchone()
  if stored:assert stored[0]==raw,'journal_provenance_changed'
  else:db.execute('INSERT INTO state VALUES (?,?)',('protocol',raw));db.commit()
  phase=db.execute('SELECT value FROM state WHERE key="band_index_complete"').fetchone()
  if not phase:
   build_band_index(db,rows,started+75)
  indexed=db.execute('SELECT value FROM state WHERE key="band_index_complete"').fetchone()
  processed=0
  query_rows=[r for r in rows if r[1] in ('train','val')];by_id={r[0]:r for r in rows}
  last=db.execute('SELECT value FROM state WHERE key="last_query_id"').fetchone();last_id=int(last[0]) if last else -1
  if indexed:
   processed,last_id,query_rows=process_queries(db,rows,retained,started+75)
  completed=sum(r[0]<=last_id for r in query_rows)
  status='VG_INTERNAL_PHASH_CANDIDATES_COMPLETE' if indexed and completed==len(query_rows) else 'VG_INTERNAL_PHASH_AUDIT_IN_PROGRESS'
  out={'utc':datetime.now(timezone.utc).isoformat(),'status':status,
       'preregistration_sha256':sha(lock),'source_hashes':hashes,'vg_source_database_sha256':before_sha,
       'band_index_complete':bool(indexed),'band_rows':db.execute('SELECT COUNT(*) FROM bands').fetchone()[0],
       'batch_queries_processed':processed,'completed_queries':completed,'target_queries':len(query_rows),
       'candidate_pairs':db.execute('SELECT COUNT(*) FROM pairs').fetchone()[0],
       'retained_draft_source_id_count':db.execute('SELECT COUNT(DISTINCT source_id) FROM pairs WHERE retained_draft=1').fetchone()[0],
       'batch_wall_seconds':time.monotonic()-started,'journal_bytes':journal.stat().st_size,
       'journal_source_unchanged':sha(source_path)==before_sha,'external_coverage_complete':False,
       'training_manifest_modified':False,'model_or_caption_reference_used':False,'test_scored':False,'clean_release_certified':False,
       'note':'Near candidates require conservative new-release exclusions; pHash similarity is not proof of same-photo identity; partial query count is not complete audit'}
  assert out['journal_source_unchanged'] and out['journal_bytes']<=128*1024**2
  if status.endswith('_COMPLETE'):
   exclusions=db.execute('SELECT DISTINCT source_id FROM pairs WHERE retained_draft=1 ORDER BY source_id').fetchall()
   evidence={'protocol':protocol,'status':status,'retained_draft_near_candidate_exclusion_ids':[r[0] for r in exclusions],
             'pairs':db.execute('SELECT * FROM pairs ORDER BY source_id,target_id').fetchall(),
             'training_manifest_modified':False,'test_scored':False,'clean_release_certified':False}
   evidence_raw=json.dumps(evidence,indent=2).encode();digest=hashlib.sha256(evidence_raw).hexdigest()
   path=folder/('evidence_'+digest+'.json')
   if path.exists():assert path.read_bytes()==evidence_raw
   else:path.write_bytes(evidence_raw)
   out.update(evidence_sha256=digest,evidence_relative_path=str(path.relative_to(root)))
 (root/'reports/VG_INTERNAL_PHASH_PROGRESS.json').write_text(json.dumps(out,indent=2))
 signal.alarm(0);print(json.dumps(out))

if __name__=='__main__':
 try:main()
 except Exception as exc:
  signal.alarm(0)
  out={'utc':datetime.now(timezone.utc).isoformat(),'status':'VG_INTERNAL_PHASH_AUDIT_BLOCKED',
       'error_type':type(exc).__name__,'error':str(exc),'clean_release_certified':False,
       'training_manifest_modified':False,'test_scored':False}
  (Path.home()/'rsg-former/reports/VG_INTERNAL_PHASH_PROGRESS.json').write_text(json.dumps(out,indent=2))
  print(json.dumps(out))
