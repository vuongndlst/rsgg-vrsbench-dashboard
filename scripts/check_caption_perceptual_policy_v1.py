"""CPU-only contracts for held-out directions and exact Hamming retrieval."""
import sys,pathlib,json,hashlib,itertools,random
root=pathlib.Path(__file__).absolute().parents[1];sys.path.insert(0,str(root/'src'))
from rsgformer.caption_phash_candidates_v1 import BANDS,bands,distance,rotation_match,protected_direction,value
enc=lambda n:format(n,'016x')
assert sum(w for _,w in BANDS)==63
base=0x18490312bc556701;count=0
for size in range(5):
 for positions in itertools.combinations(range(16),size):
  changed=base
  for bit in positions:changed^=1<<bit
  assert set(bands(enc(base))) & set(bands(enc(changed)))
  assert distance(enc(base),enc(changed))==size
  assert rotation_match([enc(base)]*4,[enc(changed)]*4)[0]==size
  count+=1
r=random.Random(20261011)
for _ in range(200):
 a=r.randrange(1<<63);b=a
 for bit in r.sample(range(63),4):b^=1<<bit
 assert set(bands(enc(a))) & set(bands(enc(b))) and distance(enc(a),enc(b))==4
outside=sum(1<<p for p in [0,13,26,39,51])
assert rotation_match([enc(0)]*4,[enc(outside)]*4) is None
for split,target in [('train','val'),('train','test'),('val','test')]:assert protected_direction(split,'vg',target)
for split,target in [('test','train'),('val','train'),('train','train')]:assert not protected_direction(split,'vg',target)
for dataset in ['coco_karpathy','flickr30k_karpathy','openimages_test']:
 assert protected_direction('train',dataset,'test') and protected_direction('val',dataset,'val')
 assert not protected_direction('test',dataset,'test') and not protected_direction('train',dataset,'train')
assert not protected_direction('train','unknown','test')
assert rotation_match([enc(outside)]*3+[enc(base)],[enc(base)]*4)[1]==270
for bad in ['8000000000000000','z'*16,'0000']:
 try:value(bad)
 except ValueError:pass
 else:raise AssertionError('bad hash accepted')
sources={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in ['src/rsgformer/caption_phash_candidates_v1.py','configs/caption_perceptual_exclusion_v1.json','scripts/check_caption_perceptual_policy_v1.py']}
out={'status':'PASS','synthetic_exhaustive_16bit_masks':count,'synthetic_full_63bit_fourflip_cases':200,
     'source_hashes':sources,'protected_dataset_contract':'canonical journal names, not display names',
     'no_network_or_pixel_access':True,'test_scored':False,'candidate_journal_launched':False,
     'final_manifest_generated':False,'clean_release_certified':False}
(root/'reports/CAPTION_PERCEPTUAL_POLICY_FIXTURE_V1.json').write_text(json.dumps(out,indent=2),encoding='utf8')
print(json.dumps(out))
