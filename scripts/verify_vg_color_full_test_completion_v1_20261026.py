"""Local byte/provenance and SG-sidecar audit of committed outputs, no inference/scoring."""
from pathlib import Path
from datetime import datetime,timezone
import collections,gzip,hashlib,importlib.util,itertools,json,zipfile
ROOT=Path(__file__).resolve().parents[1];REPORTS=ROOT/'reports';CACHE=ROOT/'artifacts/vg_color_full_test_auto_v1_20261025'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
metadata=json.loads((REPORTS/'VG_COLOR_AUTO_FULL_COMPLETION_EXPORT_20261026.json').read_text(encoding='utf8'))
assert sha(REPORTS/'VG_COLOR_AUTO_FULL_COMPLETION_EVIDENCE_20261026.zip')==metadata['archive_sha256']
with zipfile.ZipFile(REPORTS/'VG_COLOR_AUTO_FULL_COMPLETION_EVIDENCE_20261026.zip') as z:
 for name,h in metadata['files_sha256'].items():assert hashlib.sha256(z.read(name)).hexdigest()==h
 complete=json.loads(z.read('color_cache_complete.json'));metric=json.loads(z.read('TEST_diagnostic_complete.json'));cfg=json.loads(z.read('config.json'))
 lock=json.loads(z.read('pipeline_preregistration.json'));pipeline=json.loads(z.read('pipeline_complete.json'));replay=json.loads(z.read('VAL_replay_complete.json'))
 assert replay['nodes']==3083 and replay['difference_cases']==0 and replay['maximum_numeric_difference']==0
 for p,h in lock['source_sha256'].items():
  if (ROOT/p).exists():assert sha(ROOT/p)==h,('locked_source_drift',p)
 for name in ['candidate_records.jsonl.gz','full_sg_color_candidates.jsonl.gz']:assert sha(CACHE/name)==complete['output_sha256'][name]
 policy=ROOT/'src/rsgformer/vg_color_full_test_policy_v1.py';spec=importlib.util.spec_from_file_location('frozen_policy',policy);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 rows={};counts=collections.Counter();eligible=set(cfg['eligible_classes'])
 with gzip.open(CACHE/'candidate_records.jsonl.gz','rt',encoding='utf8') as f:
  for row in map(json.loads,f):
   assert row['key'] not in rows
   binding=bool(row.get('native_object_binding') and row['native_object_binding']['profile_pass']['strict'])
   colors=row['methods']['SAM2_ColorNames'].get('operating_points',{}).get('conservative',{}).get('colors',[]) if binding else []
   rows[row['key']]=(row['image_id'],row['node']['id'],colors)
   counts['requested_nodes']+=1
 assert counts['requested_nodes']==88345
 commits=iter(map(json.loads,z.read('color_image_commits.jsonl').decode('utf8').splitlines()));seen=set();used=set()
 with gzip.open(CACHE/'full_sg_color_candidates.jsonl.gz','rt',encoding='utf8') as f:
  for text,commit in itertools.zip_longest(f,commits):
   assert text is not None and commit is not None
   assert hashlib.sha256(text.rstrip('\n').encode('utf8')).hexdigest()==commit['SG_row_sha256']
   row=json.loads(text);i=row['image_id'];graph=row['graph'];annotations=row['color_annotations']
   assert i==commit['image_id'] and i not in seen;seen.add(i);counts['images']+=1
   assert row['prediction_source_sha256']==cfg['TEST_prediction_sha256'] and not row['human_precision_gate_passed']
   nodes={n['id']:n for n in graph['nodes']};assert set(annotations)==set(nodes)
   selected={n['id'] for n in mod.select_nodes(graph,eligible,cfg['maximum_requested_nodes_per_image'])}
   assert len(selected)==commit['requested_nodes']<=4
   assert all(e['subject'] in nodes and e['object'] in nodes for e in graph['edges'])
   for node_id,annotation in annotations.items():
    assert 'accepted_colors' not in nodes[node_id]
    if node_id not in selected:assert annotation['status']=='not_requested'
    else:
     key=str(i)+':'+node_id;identity,original_node,colors=rows[key];assert identity==i and original_node==node_id
     assert annotation['candidate_colors']==colors and annotation['human_verified'] is False
     assert annotation['status']==('candidate_not_human_validated' if colors else 'unknown');used.add(key)
    counts['node_'+annotation['status']]+=1
 assert seen and len(seen)==26446 and len(used)==len(rows)
 assert dict(counts)=={k:v for k,v in complete['counts'].items() if k in counts}
 assert counts['node_candidate_not_human_validated']==metric['counts']['candidate_nodes']==5214
 result={'status':'PASS_LOCAL_COMPLETION_SOURCE_LEDGER_CACHE_AND_FULL_SG_STRUCTURE','utc':datetime.now(timezone.utc).isoformat(),
  'archive_sha256':metadata['archive_sha256'],'pipeline_preregistration_sha256':pipeline['preregistration_sha256'],
  'color_cache_completion_sha256':metadata['files_sha256']['color_cache_complete.json'],
  'diagnostic_completion_sha256':metadata['files_sha256']['TEST_diagnostic_complete.json'],
  'local_binary_cache_SHA_verified':True,'all_SG_row_commit_hashes_verified':True,'all_original_graph_nodes_have_explicit_color_status':True,
  'predicted_only_registered_selection_and_strict_binding_verified':True,'counts':dict(counts),
  'VAL_full_replay_numeric_max_difference':0,'human_precision_gate_passed':False,'human_labels':0,
  'SGDet_inference_or_metric_rerun':False,'TEST_pixels_or_GT_read_by_this_audit':False}
 (REPORTS/'VG_COLOR_AUTO_FULL_COMPLETION_VERIFICATION_20261026.json').write_text(json.dumps(result,indent=2),encoding='utf8')
 print(json.dumps(result))
