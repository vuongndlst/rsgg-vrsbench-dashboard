"""Separate TRAIN/VAL attribute diagnostics; no GT attributes in generation cache."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os,re,sqlite3
import h5py,numpy as np
ROOT=Path.home()/'rsg-former';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
targets=json.loads((BASE/'color_candidates_v2/target_node_manifest.json').read_text());outdir=BASE/'color_gt_matches_v1';assert not outdir.exists();outdir.mkdir()
info=json.loads((ROOT/'data/vg150/VG-SGG-dicts-with-attri.json').read_text());attrs={int(k):v for k,v in info['idx_to_attribute'].items()}
metadata=[i for i in json.loads((ROOT/'data/vg150/image_data.json').read_text()) if i['image_id'] not in {1592,1722,4616,4617}]
indices={r['image_id']:i for i,r in enumerate(metadata)}
core=set('black white red green blue yellow brown gray orange pink purple beige tan maroon cream khaki silver gold blond'.split());aliases={'grey':'gray','golden':'gold','blonde':'blond'}
def colors(text):return sorted({aliases.get(w,w) for w in re.findall('[a-z]+',text.lower()) if aliases.get(w,w) in core})
byimage={}
for t in targets:byimage.setdefault(t['image_id'],[]).append(t)
def iou(a,b):
 inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])+1).prod(1)
 aa=np.maximum(0,a[2:]-a[:2]+1).prod();bb=np.maximum(0,b[:,2:]-b[:,:2]+1).prod(1)
 return inter/np.maximum(1e-12,aa+bb-inter)
counts={'objects':0,'matched':0,'matched_with_explicit_color':0};rows=[]
with h5py.File(ROOT/'data/vg150/VG-SGG-with-attri.h5','r') as f:
 for identity,local in byimage.items():
  assert all(t['partition'] in ['train_bank','caption_dev','caption_check','color_dev','color_check'] for t in local)
  index=indices[identity];m=metadata[index];start=int(f['img_to_first_box'][index]);end=int(f['img_to_last_box'][index])+1
  boxes=f['boxes_1024'][start:end].copy();boxes[:,:2]=boxes[:,:2]-boxes[:,2:]/2;boxes[:,2:]=boxes[:,:2]+boxes[:,2:]
  boxes=boxes.astype(float)*max(m['width'],m['height'])/1024;boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,m['width']-1);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,m['height']-1)
  labels=f['labels'][start:end,0];attributes=f['attributes'][start:end];used=set()
  for t in sorted(local,key=lambda t:(-t['node']['object_score'],t['key'])):
   node=t['node'];overlap=iou(np.asarray(node['bbox_xyxy']),boxes);choices=[i for i in np.argsort(-overlap) if overlap[i]>=.5 and labels[i]==node['class_id'] and i not in used]
   row={'key':t['key'],'image_id':identity,'partition':t['partition'],'class':node['label'],'gt_match':bool(choices),'color_labels':[],'gt_missing_color_is_unlabeled':True};counts['objects']+=1
   if choices:
    i=int(choices[0]);used.add(i);row['gt_h5_object_id']=start+i;row['match_iou']=float(overlap[i]);raw=[attrs[int(v)] for v in attributes[i] if int(v)>0]
    row['color_labels']=sorted({c for a in raw for c in colors(a)});counts['matched']+=1;counts['matched_with_explicit_color']+=bool(row['color_labels'])
   rows.append(row)
with gzip.open(outdir/'color_gt_diagnostics_separate.jsonl.gz','xt',encoding='utf-8') as f:
 for row in rows:f.write(json.dumps(row)+'\n')
# Exclude TRAIN demonstrations with exact pixels shared with protected VG VAL/TEST.
with sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
 protected={x[0] for x in db.execute("SELECT rgb_sha FROM pixels WHERE split!='train'")}
 bank=json.loads((BASE/'membership.json').read_text())['train_bank'];blocked=[i for i in bank if db.execute('SELECT rgb_sha FROM pixels WHERE vg_id=?',(i,)).fetchone()[0] in protected]
(outdir/'protected_train_bank_ids.json').write_text(json.dumps(blocked))
result={'status':'TRAIN_VAL_COLOR_GT_MATCHING_COMPLETE_SCORER_ONLY','utc':datetime.now(timezone.utc).isoformat(),'counts':counts,
 'blocked_train_bank_exact_pixel_duplicate_ids':blocked,'test_color_attributes_read':False,'gt_attributes_passed_to_generator':False,'test_reference_text_read':False}
(outdir/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
