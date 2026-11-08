"""Small immutable launch/VAL diagnostic evidence export; no TEST or active stream reads."""
from pathlib import Path
import hashlib,json,shutil,zipfile
root=Path.home()/'rsg-former';phase=root/'artifacts/vg_color_auto_full_pipeline_v1_20261025'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(phase/'preregistration.json')=='2f732c0e678d20f3023582a43201d6d99712c4430579fab79595e7d6e3b58339'
assert shutil.disk_usage(root).free>=16*1024**3+1024**2
lock=json.loads((phase/'preregistration.json').read_text())
for p,h in lock['source_sha256'].items():assert sha(root/p)==h
output=root/'reports/VG_COLOR_AUTO_FULL_INITIAL_EVIDENCE_20261025.zip';assert not output.exists()
files={
 'pipeline_preregistration.json':phase/'preregistration.json',
 'launch.json':root/'reports/VG_COLOR_AUTO_FULL_LAUNCH_V1_20261025.json',
 'status_snapshot.json':phase/'status.json',
 'config.json':root/'configs/vg_color_automated_full_test_v1_20261025.json',
 'source_manifest.json':root/'configs/vg_color_auto_full_source_manifest_v1_20261025.json',
 'VAL_diagnostic_preregistration.json':root/'artifacts/vg_color_auto_val_diagnostic_v1_20261025/preregistration.json',
 'VAL_diagnostic_complete.json':root/'artifacts/vg_color_auto_val_diagnostic_v1_20261025/complete.json',
 'VAL_diagnostic_details.jsonl.gz':root/'artifacts/vg_color_auto_val_diagnostic_v1_20261025/diagnostic/matched_attribute_diagnostics.jsonl.gz',
 'VAL_replay_preregistration.json':root/'artifacts/vg_color_fast_val_replay_v1_20261025/preregistration.json'}
assert sum(p.stat().st_size for p in files.values())<1024**2
with zipfile.ZipFile(output,'x',zipfile.ZIP_DEFLATED) as z:
 for name,p in files.items():z.write(p,name)
result={'status':'INITIAL_COLOR_PHASE_SOURCE_AND_VAL_DIAGNOSTIC_EVIDENCE_EXPORTED','archive':str(output.relative_to(root)),
 'archive_sha256':sha(output),'files_sha256':{name:sha(p) for name,p in files.items()},'TEST_pixels_or_attributes_read':False,'active_inference_stream_not_read':True}
(root/'reports/VG_COLOR_AUTO_FULL_INITIAL_EXPORT_20261025.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
