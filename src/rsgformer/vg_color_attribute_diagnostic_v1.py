"""Frozen weak-attribute diagnostic, isolated from extraction and never called as GT input."""
from pathlib import Path
import collections,gzip,hashlib,itertools,json,re
import h5py,numpy as np

CORE=set('black white gray red green blue yellow brown orange pink purple'.split())
EXTENDED=CORE|set('beige tan maroon cream khaki silver gold blond'.split())
ALIASES={'grey':'gray','golden':'gold','blonde':'blond'}
def labels(text):return sorted({ALIASES.get(w,w) for w in re.findall('[a-z]+',text.lower()) if ALIASES.get(w,w) in EXTENDED})
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def inclusive_iou(a,b):
 inter=np.maximum(0,np.minimum(a[2:],b[:,2:])-np.maximum(a[:2],b[:,:2])+1).prod(1)
 aa=np.maximum(0,a[2:]-a[:2]+1).prod();bb=np.maximum(0,b[:,2:]-b[:,:2]+1).prod(1)
 return inter/np.maximum(1e-12,aa+bb-inter)
def proposed_colors(row):
 colors=row['methods']['SAM2_ColorNames'].get('operating_points',{}).get('conservative',{}).get('colors',[])
 passed=bool(row.get('native_object_binding') and row['native_object_binding']['profile_pass']['strict'])
 return colors if passed else []
def evaluate(root,records_path,out_dir,expected_image_ids,target_manifest=None,expected_split='val',progress=None):
 root=Path(root);out_dir=Path(out_dir);out_dir.mkdir(exist_ok=False)
 info=json.loads((root/'data/vg150/VG-SGG-dicts-with-attri.json').read_text());attributes={int(k):v for k,v in info['idx_to_attribute'].items()}
 class_ids={v:int(k) for k,v in info['idx_to_label'].items()}
 metadata=[m for m in json.loads((root/'data/vg150/image_data.json').read_text()) if m['image_id'] not in {1592,1722,4616,4617}];indices={m['image_id']:i for i,m in enumerate(metadata)}
 targets={t['key']:t['node'] for t in json.loads(Path(target_manifest).read_text())} if target_manifest else None
 expected=set(expected_image_ids);seen=set();counts=collections.Counter();byclass={};details=out_dir/'matched_attribute_diagnostics.jsonl.gz'
 with h5py.File(root/'data/vg150/VG-SGG-with-attri.h5','r') as f,gzip.open(records_path,'rt') as inp,gzip.open(details,'xt') as out:
  stream=map(json.loads,inp)
  for identity,group in itertools.groupby(stream,key=lambda r:r['image_id']):
   assert identity in expected and identity not in seen;seen.add(identity);local=list(group);meta=metadata[indices[identity]];idx=indices[identity]
   assert int(f['split'][idx])==(2 if expected_split=='test' else 0)
   start=int(f['img_to_first_box'][idx]);end=int(f['img_to_last_box'][idx])+1;assert start>=0
   boxes=f['boxes_1024'][start:end].copy();boxes[:,:2]=boxes[:,:2]-boxes[:,2:]/2;boxes[:,2:]=boxes[:,:2]+boxes[:,2:]
   boxes=boxes.astype(np.float32);boxes*=max(meta['width'],meta['height'])/1024
   boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,meta['width']-1);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,meta['height']-1)
   gtlabels=f['labels'][start:end,0];gtattrs=f['attributes'][start:end];used=set()
   def node(r):return targets[r['key']] if targets is not None else r['node']
   for row in sorted(local,key=lambda r:(-node(r)['object_score'],r['key'])):
    n=node(row);prediction=proposed_colors(row);class_id=int(n.get('class_id',class_ids[row['class']]))
    overlap=inclusive_iou(np.asarray(n['bbox_xyxy']),boxes);eligible=[int(j) for j in np.argsort(-overlap,kind='stable') if overlap[j]>=.5 and gtlabels[j]==class_id and int(j) not in used]
    item={'key':row['key'],'image_id':identity,'class':row['class'],'predicted_color_candidates':prediction,'matched_same_class_IoU05':bool(eligible),
     'GT_missing_is_unlabeled':True,'human_annotation':False,'GT_labels':[]}
    c=byclass.setdefault(row['class'],collections.Counter());counts['selected_nodes']+=1;c['selected_nodes']+=1
    if prediction:counts['candidate_nodes']+=1;c['candidate_nodes']+=1
    if eligible:
     j=eligible[0];used.add(j);raw=[attributes[int(a)] for a in gtattrs[j] if int(a)>0];truth=sorted({v for a in raw for v in labels(a)})
     item.update(GT_labels=truth,match_iou=float(overlap[j]),GT_object_id=start+j)
     counts['same_class_matched_nodes']+=1;c['same_class_matched_nodes']+=1
     if truth:
      counts['matched_explicit_color_nodes']+=1;c['matched_explicit_color_nodes']+=1
      if prediction:
       counts['candidate_nodes_with_explicit_GT_color']+=1;c['candidate_nodes_with_explicit_GT_color']+=1
       subset=set(prediction)<=set(truth);exact=set(prediction)==set(truth)
       counts['prediction_subset_of_explicit_GT']+=subset;c['prediction_subset_of_explicit_GT']+=subset
       counts['prediction_exact_explicit_GT_set']+=exact;c['prediction_exact_explicit_GT_set']+=exact
       if set(truth)<=CORE:
        counts['candidate_nodes_with_core11_only_GT']+=1;c['candidate_nodes_with_core11_only_GT']+=1
        counts['core11_subset_agreement_count']+=subset;c['core11_subset_agreement_count']+=subset
       else:counts['candidate_nodes_with_extended_GT_terms']+=1;c['candidate_nodes_with_extended_GT_terms']+=1
     else:counts['matched_color_unlabeled_nodes']+=1
    else:counts['no_same_class_match_nodes']+=1
    out.write(json.dumps(item)+'\n')
   if progress and len(seen)%500==0:progress(len(seen),dict(counts))
 assert counts['selected_nodes']>=counts['candidate_nodes']
 denom=counts['candidate_nodes_with_explicit_GT_color'];coredenom=counts['candidate_nodes_with_core11_only_GT']
 result={'status':'AUTOMATED_VG_ATTRIBUTE_DIAGNOSTIC_COMPLETE_NOT_HUMAN_PRECISION','split':expected_split,'expected_images':len(expected),'images_with_requested_nodes':len(seen),
  'images_without_requested_nodes_retained':len(expected)-len(seen),'counts':dict(counts),'by_class':{k:dict(v) for k,v in byclass.items()},
  'candidate_coverage':counts['candidate_nodes']/max(1,counts['selected_nodes']),
  'explicit_positive_GT_subset_agreement':counts['prediction_subset_of_explicit_GT']/denom if denom else None,
  'explicit_positive_GT_exact_set_agreement':counts['prediction_exact_explicit_GT_set']/denom if denom else None,
  'core11_positive_GT_subset_agreement':counts['core11_subset_agreement_count']/coredenom if coredenom else None,
  'diagnostic_details_sha256':sha(details),'human_labels_completed':0,'human_precision_gate_passed':False,
  'limitations':['VG attributes are partial weak labels','unmatched nodes are not automatically false positives','missing color is unlabeled','agreement on positive matched labels is not complete visual accuracy','extended colors are not silently remapped','one-to-one matching is conditional on predicted classes and IoU'],
  'matching':'native H5 integer center/half-size truncation, float32 original-image scaling, clipped inclusivexyxy IoU>=.5; descending predicted score then key',
  'source_sha256':sha(Path(__file__))}
 (out_dir/'complete.json').write_text(json.dumps(result,indent=2));return result
