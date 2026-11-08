"""Read-only completion audit and bounded evidence export; never rerun inference/metrics."""
from pathlib import Path
import collections,gzip,hashlib,json,os,resource,shutil,zipfile
root=Path.home()/'rsg-former';pipe=root/'artifacts/vg_color_auto_full_pipeline_v1_20261025'
cache=root/'artifacts/vg_color_full_test_auto_v1_20261025';metric=root/'artifacts/vg_color_auto_test_diagnostic_v1_20261025';replay=root/'artifacts/vg_color_fast_val_replay_v1_20261025'
os.sched_setaffinity(0,{sorted(os.sched_getaffinity(0))[0]})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
assert sha(pipe/'preregistration.json')=='2f732c0e678d20f3023582a43201d6d99712c4430579fab79595e7d6e3b58339'
lock=json.loads((pipe/'preregistration.json').read_text());terminal=json.loads((pipe/'complete.json').read_text());result=json.loads((cache/'complete.json').read_text());diagnostic=json.loads((metric/'complete.json').read_text())
assert terminal['status']=='COMPLETE_AUTOMATED_COLOR_TEST_NOT_HUMAN_PRECISION'
for stage in terminal['completed_stages']:
 folder={'VAL_PARITY':replay,'FULL_TEST_COLOR':cache,'TEST_ATTRIBUTE_DIAGNOSTIC':metric}[stage['stage']]
 assert sha(folder/'complete.json')==stage['completion_sha256']
for p,h in lock['source_sha256'].items():assert sha(root/p)==h,('source_drift',p)
assert sha(root/'configs/vg_color_automated_full_test_v1_20261025.json')==lock['config_sha256']
for p,h in result['output_sha256'].items():assert sha(cache/p)==h,('cache_hash',p)
assert sha(metric/'diagnostic/matched_attribute_diagnostics.jsonl.gz')==diagnostic['diagnostic_details_sha256']
assert sha(root/'artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/best.pt')=='29017a69d7cb6d76d68c834b5d540712775a9e0cf9cae13efd17d78a7b69292e'
assert diagnostic['full_color_cache_completion_sha256']==sha(cache/'complete.json')
assert not result['GT_accessed_by_extractor'] and not result['new_SGDet_inference_or_scoring'] and not result['caption_generated']
ledger=json.loads((cache/'test_access.json').read_text());assert ledger['status']=='OPENED_EXCLUSIVE_COLOR_ACCESS_NO_RETRY' and not ledger['GT_access_allowed']
ids=[];node_counts=collections.Counter()
with (cache/'image_commits.jsonl').open() as f:
 for row in map(json.loads,f):ids.append(row['image_id']);node_counts['requested_nodes']+=row['requested_nodes'];assert row['source_index']==len(ids)-1
assert len(ids)==len(set(ids))==26446 and node_counts['requested_nodes']==88345
counts=collections.Counter();last_image=None;seen=set()
with gzip.open(cache/'candidate_records.jsonl.gz','rt') as f:
 for row in map(json.loads,f):
  identity=row['image_id'];assert row['key']==str(identity)+':'+row['node']['id'] and row['class']==row['node']['label']
  if identity!=last_image:assert identity not in seen;seen.add(identity);last_image=identity
  counts['records']+=1
assert counts['records']==result['counts']['requested_nodes']==diagnostic['counts']['selected_nodes']==88345
assert len(seen)==result['counts']['images_with_requested_nodes']==24386
assert len(ids)-len(seen)==result['counts']['images_without_requested_nodes']==2060
assert result['counts']['node_unknown']+result['counts']['node_candidate_not_human_validated']==88345
assert diagnostic['counts']['candidate_nodes']==result['counts']['node_candidate_not_human_validated']==5214
assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<4*1024**3
files={'pipeline_preregistration.json':pipe/'preregistration.json','pipeline_complete.json':pipe/'complete.json',
 'VAL_replay_complete.json':replay/'complete.json','VAL_replay_preregistration.json':replay/'preregistration.json',
 'color_cache_preregistration.json':cache/'preregistration.json','color_cache_complete.json':cache/'complete.json','color_TEST_access.json':cache/'test_access.json',
 'color_image_commits.jsonl':cache/'image_commits.jsonl','TEST_diagnostic_preregistration.json':metric/'preregistration.json','TEST_diagnostic_complete.json':metric/'complete.json',
 'TEST_diagnostic_details.jsonl.gz':metric/'diagnostic/matched_attribute_diagnostics.jsonl.gz','config.json':root/'configs/vg_color_automated_full_test_v1_20261025.json',
 'source_manifest.json':root/'configs/vg_color_auto_full_source_manifest_v1_20261025.json'}
assert sum(p.stat().st_size for p in files.values())<=16*1024**2
assert shutil.disk_usage(root).free>=16*1024**3+16*1024**2
output=root/'reports/VG_COLOR_AUTO_FULL_COMPLETION_EVIDENCE_20261026.zip';assert not output.exists(),'immutable_export_no_overwrite'
with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
report={'status':'PASS_COMPLETE_SOURCE_CHECKPOINT_LEDGER_CACHE_AND_DIAGNOSTIC_EVIDENCE',
 'archive':str(output.relative_to(root)),'archive_sha256':sha(output),'archive_bytes':output.stat().st_size,
 'files_sha256':{name:sha(p) for name,p in files.items()},'cache_files':{p:{'sha256':h,'bytes':(cache/p).stat().st_size} for p,h in result['output_sha256'].items()},
 'source_hashes_verified':True,'checkpoint_hash_verified':True,'full_denominator_verified':True,'cached_nodes_verified':88345,
 'human_precision_gate_passed':False,'human_labels':0,'TEST_inference_or_metric_rerun':False,'new_TEST_pixels_or_GT_read_by_this_audit':False}
(root/'reports/VG_COLOR_AUTO_FULL_COMPLETION_EXPORT_20261026.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
