"""CPU SQLite contract fixture; no real images, remote access or journal edits."""
import ast,hashlib,json,random,sqlite3,time
from pathlib import Path
from types import SimpleNamespace
import sys
root=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(root/'src'))
source=root/'scripts/remote_vg_internal_phash_batch_v1.py'
tree=ast.parse(source.read_text())
selected=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in
          ('BatchDeadline','check_deadline','candidate_ids','build_band_index','process_queries')]
scope={'time':time}
exec(compile(ast.Module(body=selected,type_ignores=[]),str(source),'exec'),scope)
def database():
 db=sqlite3.connect(':memory:')
 db.executescript('CREATE TABLE state(key TEXT PRIMARY KEY,value TEXT); CREATE TABLE bands(band INTEGER,value INTEGER,vg_id INTEGER,rotation INTEGER,PRIMARY KEY(band,value,vg_id,rotation)) WITHOUT ROWID; CREATE TABLE pairs(source_id INTEGER,target_id INTEGER,distance INTEGER,source_rotation INTEGER,target_rotation INTEGER,retained_draft INTEGER,PRIMARY KEY(source_id,target_id)) WITHOUT ROWID;')
 return db
def enc(n):return format(n,'016x')
hs=lambda n:[enc(n)]*4
rows=[(1,'train',*hs(0)),(2,'train',*hs(1)),(3,'val',*hs(3)),(4,'test',*hs(7)),
      (5,'test',*hs(sum(1<<offset for offset in (0,13,26,39,51))))]
retained={'train':{1},'val':{3},'test':{4,5}}
db=database()
# Interrupt band construction after exactly one protected identity, then resume.
clock=iter([0,2])
scope['time']=SimpleNamespace(monotonic=lambda:next(clock))
scope['build_band_index'](db,rows,1)
assert db.execute('SELECT value FROM state WHERE key="last_band_id"').fetchone()[0]=='3'
assert db.execute('SELECT COUNT(*) FROM bands').fetchone()[0]==20
scope['time']=time
scope['build_band_index'](db,rows,time.monotonic()+10)
assert db.execute('SELECT COUNT(*) FROM bands').fetchone()[0]==60
assert db.execute('SELECT value FROM state WHERE key="band_index_complete"').fetchone()
assert 5 not in scope['candidate_ids'](db,hs(0))  # five distinct band flips, outside radius4
processed,last,_=scope['process_queries'](db,rows,retained,time.monotonic()+10,limit=1)
assert (processed,last)==(1,1)
assert db.execute('SELECT source_id,target_id,distance,retained_draft FROM pairs ORDER BY source_id,target_id').fetchall()==[(1,3,2,1),(1,4,3,1)]
processed,last,_=scope['process_queries'](db,rows,retained,time.monotonic()+10)
assert (processed,last)==(2,3)
assert db.execute('SELECT source_id,target_id FROM pairs ORDER BY source_id,target_id').fetchall()==[(1,3),(1,4),(2,3),(2,4),(2,5),(3,4)]
assert scope['process_queries'](db,rows,retained,time.monotonic()+10)[0]==0
# Capacity failure must roll back all pairs for an unfinished query and its marker.
fresh=database();scope['build_band_index'](fresh,rows,time.monotonic()+10)
try:scope['process_queries'](fresh,rows,retained,time.monotonic()+10,pair_cap=1)
except AssertionError as exc:assert str(exc)=='pair_cap_no_identity_omission'
else:raise AssertionError('pair_cap_not_enforced')
assert fresh.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==0
assert fresh.execute('SELECT value FROM state WHERE key="last_query_id"').fetchone() is None
# Timeout after one insert must have exactly the same rollback semantics.
old_check=scope['check_deadline'];calls=[0]
def interrupt(deadline):
 calls[0]+=1
 if calls[0]==7:raise scope['BatchDeadline']()
scope['check_deadline']=interrupt
assert scope['process_queries'](fresh,rows,retained,time.monotonic()+10)[0]==0
assert fresh.execute('SELECT COUNT(*) FROM pairs').fetchone()[0]==0
assert fresh.execute('SELECT value FROM state WHERE key="last_query_id"').fetchone() is None
scope['check_deadline']=old_check
from rsgformer.caption_phash_candidates_v1 import rotation_match,protected_direction
rng=random.Random(4455)
random_rows=[(i,'train' if i<11 else 'val' if i<16 else 'test',*[enc(rng.getrandbits(63)) for _ in range(4)]) for i in range(1,21)]
# Plant four-bit matches across every source/target rotation combination.
for i in range(1,11):
 target=16+(i%5);si=i%4;ti=(i//4)%4
 row=list(random_rows[target-1]);row[2+ti]=enc(int(random_rows[i-1][2+si],16)^15)
 random_rows[target-1]=tuple(row)
rotated=database();scope['build_band_index'](rotated,random_rows,time.monotonic()+10)
scope['process_queries'](rotated,random_rows,{'train':set(),'val':set()},time.monotonic()+10)
expected=[]
for source_row in random_rows:
 for target_row in random_rows:
  if protected_direction(source_row[1],'vg',target_row[1]):
   match=rotation_match(source_row[2:],target_row[2:])
   if match:expected.append((source_row[0],target_row[0],*match,0))
assert rotated.execute('SELECT * FROM pairs ORDER BY source_id,target_id').fetchall()==sorted(expected)
report={'status':'PASS','source_hashes':{str(source.relative_to(root)).replace('\\','/'):hashlib.sha256(source.read_bytes()).hexdigest()},
        'contracts':['bounded band resume','full Hamming filter and protected direction','query resume and completion','pair cap atomic rollback','timeout atomic rollback','SQL retrieval equals all-pairs comparison with planted rotated matches'],
        'real_fingerprints_accessed':False,'clean_release_certified':False,'test_scored':False}
(root/'reports/VG_INTERNAL_PHASH_BATCH_FIXTURE_V1.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
