"""Finalize already-written deterministic membership after provenance upload repair."""
from pathlib import Path
from datetime import datetime,timezone
import collections,hashlib,json,sqlite3
root=Path.home()/'rsg-former';art=root/'artifacts/color_fresh_val_manifest_v1_20261025';base=root/'artifacts/vg_caption_execution_v1_20261025'
assert (art/'membership.json').exists() and not (art/'preregistration.json').exists()
membership=json.loads((art/'membership.json').read_text());targets=json.loads((art/'target_manifest.json').read_text());missing=json.loads((art/'images_without_target.json').read_text())
old={i for v in json.loads((base/'membership.json').read_text()).values() for i in v}
with sqlite3.connect((root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
 pixels={i:(rgb,split) for i,rgb,split in db.execute('SELECT vg_id,rgb_sha,split FROM pixels')}
chosen=membership['color_dev_v1']+membership['color_check_v1'];assert len(chosen)==len(set(chosen))==1000 and len(membership['color_dev_v1'])==200
assert not set(chosen)&old and all(pixels[i][1]=='val' for i in chosen)
assert not {pixels[i][0] for i in chosen}&{pixels[i][0] for i in old}
assert len({pixels[i][0] for i in chosen})==1000
counts=collections.Counter(r['image_id'] for r in targets);assert max(counts.values())<=4
assert set(counts)|{r['image_id'] for r in missing}==set(chosen)
for r in targets:assert r['image_id'] in membership[r['partition']] and not r['selection_has_reference_or_color_input']
source=root/'scripts/remote_prepare_color_fresh_val_manifest_v1_20261025.py';assert source.exists()
report={'status':'FRESH_COLOR_VAL_DEV_CHECK_MEMBERSHIP_LOCKED_NO_INFERENCE','utc':datetime.now(timezone.utc).isoformat(),'images':1000,
 'dev_images':200,'check_images':800,'maximum_nodes_per_image':4,'targets':len(targets),'partition_target_counts':dict(collections.Counter(r['partition'] for r in targets)),
 'images_without_eligible_target_retained':len(missing),'disjoint_all_prior_engineering_parents_and_known_exact_pixels':True,
 'known_perceptual_identity_audit_complete':False,'class_scope':sorted({r['node']['label'] for r in targets}),'TEST_attributes_or_reference_text_read':False,
 'selection_uses_predicted_inputs_only':True,'pixel_or_mask_inference_run':False,'color_policy_selected_from_this_check':False,
 'check_inference_requires_prior_locked_policy_and_resource_admission':True,'human_review_gate_pending':True,
 'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'finalizer_sha256':hashlib.sha256((root/'scripts/remote_finalize_color_fresh_val_manifest_v1_20261025.py').read_bytes()).hexdigest(),
 'preparation_repair':'Initial stdin script wrote deterministic membership but lacked uploaded source for final hash; metadata kept, source uploaded, existing memberships independently rechecked before lock; no inference or reference text was read',
 'manifest_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in art.iterdir() if p.is_file()}}
(art/'preregistration.json').write_text(json.dumps(report,indent=2));(root/'reports/VG_COLOR_FRESH_VAL_MANIFEST_20261025.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
