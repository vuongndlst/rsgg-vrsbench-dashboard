"""Bounded TRAIN/observed-VAL geometry and color-support audit; no refs or inference."""
from pathlib import Path
import collections,copy,gzip,json,math,os,resource,shutil,signal,statistics,time,traceback,zipfile
ROOT=Path.home()/'rsg-former';PILOT=ROOT/'artifacts/vg_caption_pilot100_v1_20261026';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025';ART=ROOT/'artifacts/vg_region_focus_audit_v1_20261027'
os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[0]});signal.alarm(120)
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now,artifact_bytes
from rsgformer.vg_caption_graph_v1 import digest,focus_graph

def guard():
 assert time.time()<1794670321.2414355
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=768*1024**2
 assert shutil.disk_usage(ROOT).free>=16*1024**3+2*1024**2
 assert artifact_bytes(ROOT/'artifacts')<=40*1024**3
 if ART.exists():assert artifact_bytes(ART)<=2*1024**2

def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def intersection(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def stripped(g):
 g=copy.deepcopy(g)
 for n in g['nodes']:n.pop('candidate_colors',None)
 return g

def distribution(values):
 return {'n':len(values),'min':min(values) if values else None,'median':statistics.median(values) if values else None,'max':max(values) if values else None}

def main():
 guard();assert not ART.exists(),'immutable_audit_no_repeat';ART.mkdir()
 assert sha(PILOT/'preregistration.json')=='3715c7489698cae4bdc99f41116f4214ceb5e0fa0b4e772fd57d9123efe06c15'
 parent=json.loads((PILOT/'preregistration.json').read_text())
 for p,h in parent['source_sha256'].items():assert sha(ROOT/p)==h,('preserved_source_changed',p)
 prepared=json.loads((PILOT/'prepare/complete.json').read_text());generation=json.loads((PILOT/'generation/complete.json').read_text())
 for p,h in prepared['file_sha256'].items():assert sha(PILOT/'prepare'/p)==h
 assert sha(PILOT/'generation/query_units_bounded.json')==generation['query_units_sha256']
 save(ART/'preregistration.json',{'utc':now(),'source_sha256':{p:sha(ROOT/p) for p in ['scripts/remote_vg_region_focus_audit_v1_20261027.py','src/rsgformer/vg_caption_graph_v1.py','src/rsgformer/vg_color_full_test_policy_v1.py']},
  'parent_preregistration_sha256':sha(PILOT/'preregistration.json'),'inputs':'TRAIN800regions and observedVAL100pilot regions, original saved TRAIN200/VAL5000graphs, current color records and bounded queries',
  'diagnostics':['reproduce legacy object-box-overlap>=0.5 focus before token bounding','positive intersection versus fraction of region covered, without selecting an alternative focus policy','same-class bbox continuous-IoU>=0.9 pairs are duplicate-risk flags, not instance truth','first-node edge order versus maximum object score, not correctness truth','20class/max4node/color-candidate support attribution'],
  'reference_text_attributes_images_models_read':False,'TEST_access_allowed':False,'new_policy_adopted':False,'caption_or_metric_rerun':False,'limits':{'CPU':1,'RSS_MiB':768,'wall_seconds':120,'new_artifacts_MiB':2}})
 groups={}
 for label,name in [('TRAIN','train_units.jsonl.gz'),('observed_VAL','pilot_units.jsonl.gz')]:
  with gzip.open(PILOT/'prepare'/name,'rt') as f:groups[label]=list(map(json.loads,f))
 assert len(groups['TRAIN'])==800 and len(groups['observed_VAL'])==100
 bounded={u['unit_id']:u for u in json.loads((PILOT/'generation/query_units_bounded.json').read_text())}
 units=sum(groups.values(),[]);wanted={u['parent_image_id'] for u in units};graphs={}
 base_complete=json.loads((BASE/'base_cache_complete.json').read_text())
 for name in ['train_bank_base_sg.jsonl.gz','val_base_sg.jsonl.gz']:
  assert sha(BASE/name)==base_complete['file_sha256'][name]
  with gzip.open(BASE/name,'rt') as f:
   for row in map(json.loads,f):
    if row['image_id'] in wanted:graphs[row['image_id']]=row
 assert len(wanted)==300 and set(graphs)==wanted
 candidates={}
 with gzip.open(PILOT/'prepare/candidate_records.jsonl.gz','rt') as f:
  for row in map(json.loads,f):
   passed=bool(row.get('native_object_binding') and row['native_object_binding']['profile_pass']['strict'])
   colors=row['methods']['SAM2_ColorNames'].get('operating_points',{}).get('conservative',{}).get('colors',[]) if passed else []
   candidates[row['key']]=colors
 cfg=json.loads((ROOT/'configs/vg_color_automated_full_test_v1_20261025.json').read_text());eligible=set(cfg['eligible_classes'])
 counts={};distributions={};reproduced=0
 with gzip.open(ART/'unit_diagnostics.jsonl.gz','xt') as out:
  for label,local in groups.items():
   c=collections.Counter();empty_region_coverage=[];empty_object_fraction=[]
   for u in local:
    identity=u['parent_image_id'];base=graphs[identity];g=base['graph'];region=u['crop_xyxy'];assert digest(g)==base['base_graph_sha256']==u['base_graph_sha256']
    legacy=focus_graph(g,region);assert stripped(legacy)==stripped(u['graph']),'cached_focus_not_reproduced';reproduced+=1;c['regions']+=1
    overlaps=[(intersection(n['bbox_xyxy'],region),n) for n in g['nodes']]
    positive=[(x,n) for x,n in overlaps if x>0];max_region=max([x/area(region) for x,n in positive],default=0);max_object=max([x/area(n['bbox_xyxy']) for x,n in positive],default=0)
    if not legacy['nodes']:
     c['legacy_empty_regions']+=1;c['legacy_empty_with_any_intersecting_node']+=bool(positive)
     c['legacy_empty_with_node_covering_at_least_half_region']+=max_region>=.5
     empty_region_coverage.append(max_region);empty_object_fraction.append(max_object)
    if label=='observed_VAL':
     q=bounded[u['unit_id']]['graph'];c['bounded_empty_regions']+=not bool(q['nodes']);c['nodes_removed_by_token_bounding']+=len(legacy['nodes'])-len(q['nodes'])
    else:q=legacy
    duplicate_pairs=[]
    for i,a in enumerate(q['nodes']):
     for b in q['nodes'][i+1:]:
      if a['label']!=b['label']:continue
      inter=intersection(a['bbox_xyxy'],b['bbox_xyxy']);iou=inter/max(1e-12,area(a['bbox_xyxy'])+area(b['bbox_xyxy'])-inter)
      if iou>=.9:duplicate_pairs.append([a['id'],b['id'],a['label'],iou])
    c['regions_with_same_class_IoU09_risk']+=bool(duplicate_pairs);c['same_class_IoU09_pairs']+=len(duplicate_pairs)
    if q['nodes']:
     c['nonempty_regions']+=1;c['first_node_has_lower_score_than_another']+=q['nodes'][0]['object_score']<max(n['object_score'] for n in q['nodes'])
    c['focus_nodes']+=len(q['nodes']);c['focus_edges']+=len(q['edges'])
    allowed=[n for n in q['nodes'] if n['label'] in eligible];selected=[n for n in allowed if str(identity)+':'+n['id'] in candidates];colored=[n for n in selected if candidates[str(identity)+':'+n['id']]]
    c['regions_with_eligible_color_class']+=bool(allowed);c['eligible_focus_nodes']+=len(allowed);c['requested_focus_nodes']+=len(selected);c['eligible_focus_nodes_outside_parent_color_cap']+=len(allowed)-len(selected)
    c['regions_with_requested_focus_node']+=bool(selected);c['regions_with_candidate_color']+=bool(colored);c['candidate_focus_nodes']+=len(colored)
    out.write(line({'unit_id':u['unit_id'],'partition':label,'parent_image_id':identity,'crop_xyxy':region,'focus_nodes':len(q['nodes']),'focus_edges':len(q['edges']),
     'legacy_empty':not bool(legacy['nodes']),'intersecting_full_graph_nodes':len(positive),'maximum_region_coverage_by_a_node':max_region,'maximum_object_fraction_in_region':max_object,
     'same_class_IoU09_risk_pairs':duplicate_pairs,'eligible_color_nodes':len(allowed),'requested_color_nodes':len(selected),'candidate_color_nodes':len(colored)})+'\n')
   counts[label]=dict(c);distributions[label]={'legacy_empty_maximum_region_coverage':distribution(empty_region_coverage),'legacy_empty_maximum_object_fraction':distribution(empty_object_fraction)}
   guard()
 result={'status':'REGION_FOCUS_AND_COLOR_SUPPORT_AUDIT_COMPLETE_NO_POLICY_CHANGE','utc':now(),'regions_reproduced':reproduced,'counts':counts,'distributions':distributions,
  'unit_diagnostics_sha256':sha(ART/'unit_diagnostics.jsonl.gz'),'new_policy_adopted':False,'GT_reference_or_attributes_read':False,'TEST_accessed':False,'caption_or_metric_rerun':False,
  'limitations':['box overlap is not semantic correctness','same-class overlap flags do not prove duplicate physical instances','VAL100pilot has been observed and is not a fresh heldout check','no source images or GT objects/references were used','no threshold/method chosen from TEST']}
 save(ART/'complete.json',result)
 archive=ART/'audit_evidence.zip'
 with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
  for n in ['preregistration.json','unit_diagnostics.jsonl.gz','complete.json']:z.write(ART/n,n)
 result.update(archive=str(archive.relative_to(ROOT)),archive_sha256=sha(archive));save(ROOT/'reports/VG_REGION_FOCUS_AUDIT_V1_20261027.json',result);print(line(result))
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'TEST_accessed':False})
  traceback.print_exc();raise
