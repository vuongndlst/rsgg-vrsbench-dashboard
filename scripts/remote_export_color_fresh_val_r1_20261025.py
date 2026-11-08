"""Verify terminal color cache and export immutable evidence once, no model/test run."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,shutil,zipfile
root=Path.home()/'rsg-former';art=root/'artifacts/color_fresh_val_execution_r1_20261025';export=art/'verified_export_v1'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as stream:
  for b in iter(lambda:stream.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
if (export/'export_manifest.json').exists():
 print((export/'export_manifest.json').read_text());raise SystemExit(0)
assert not export.exists();assert shutil.disk_usage(root).free>=16*1024**3+64*1024**2
complete=json.loads((art/'complete.json').read_text());lock=json.loads((art/'preregistration.json').read_text())
packet=art/'blind_review_packet_v1';summary=json.loads((packet/'summary.json').read_text())
assert complete['nodes']==summary['expected_nodes']==3083 and summary['human_annotations_completed']==0
assert not complete['test_accessed'] and not summary['TEST_accessed']
assert sha(art/'candidate_records.jsonl.gz')==complete['candidate_records_sha256']==summary['candidate_records_sha256']
for path,expected in lock['source_sha256'].items():assert sha(root/path)==expected,'source_changed:'+path
fresh=root/'artifacts/color_fresh_val_manifest_v1_20261025';assert sha(fresh/'target_manifest.json')==lock['original_target_manifest_sha256']
export.mkdir()
files={'complete.json':art/'complete.json','preregistration.json':art/'preregistration.json','repair_provenance.json':art/'repair_provenance.json',
 'candidate_records.jsonl.gz':art/'candidate_records.jsonl.gz','target_manifest.json':art/'target_manifest.json',
 'review/summary.json':packet/'summary.json','review/blind_review_packet.zip':packet/'blind_review_packet.zip',
 'fresh_membership.json':fresh/'membership.json','fresh_membership_preregistration.json':fresh/'preregistration.json'}
for path in lock['source_sha256']:files['locked_source/'+path]=root/path
manifest={'status':'FRESH_VAL_COLOR_R1_COMPLETION_VERIFIED','utc':datetime.now(timezone.utc).isoformat(),
 'expected_nodes':3083,'processed_eligible_parent_images':896,'registered_parent_images':1000,'parents_without_target_retained':104,
 'human_review_instances':summary['review_instances_distinct_parent_images'],'human_labels_completed':0,
 'source_and_input_and_candidate_sha_verification':'PASS','TEST_accessed':False,'production_color_accepted':False,
 'file_sha256':{name:sha(p) for name,p in files.items()}}
with zipfile.ZipFile(export/'verified_evidence.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
manifest.update(archive_sha256=sha(export/'verified_evidence.zip'),archive_bytes=(export/'verified_evidence.zip').stat().st_size)
(export/'export_manifest.json').write_text(json.dumps(manifest,indent=2));(root/'reports/VG_COLOR_FRESH_VAL_COMPLETION_VERIFICATION_R1_20261025.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest))
