"""Read-only audit of the failed JSON export; preserve original outcome/source."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,hashlib,json,os
import numpy as np
from rsgformer.vg_object_color_rules_v3 import clipped_box
root=Path.home()/'rsg-former';art=root/'artifacts/color_fresh_val_execution_v1_20261025'
failure=json.loads((art/'failure.json').read_text());assert failure['reason']=='Object of type float32 is not JSON serializable'
controller=json.loads((root/'reports/VG_COLOR_FRESH_VAL_CONTROLLER_20261025.json').read_text());assert controller['status']=='FAILED_NO_RETRY'
with gzip.open(art/'candidate_records.jsonl.gz','rt') as f:rows=list(map(json.loads,f))
targets=json.loads((art/'target_manifest.json').read_text());target=targets[len(rows)]
with gzip.open(root/'artifacts/vg_caption_execution_v1_20261025/val_base_sg.jsonl.gz','rt') as f:
 record=next(r for r in map(json.loads,f) if r['image_id']==target['image_id'])
# Check dimensions from identity journal; image bytes and model outputs are not reopened.
import sqlite3
with sqlite3.connect((root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
 W,H=db.execute('SELECT width,height FROM pixels WHERE vg_id=?',(target['image_id'],)).fetchone()
b,visible=clipped_box(np.asarray(target['node']['bbox_xyxy'],dtype=np.float32),W,H)
result={'utc':datetime.now(timezone.utc).isoformat(),'status':'FAILED_JSON_EXPORT_AUDITED_NO_SCIENTIFIC_CHANGE','failure':failure,
 'complete_rows_preserved':len(rows),'first_uncommitted_key':target['key'],'clipped_box_visible_fraction_type':type(visible).__name__,
 'clipped_box_visible_fraction_value':float(visible),'old_pid_present':{str(p):(Path('/proc')/str(p)).exists() for p in [1546074,1546075]},
 'evidence_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [art/'failure.json',art/'preregistration.json',art/'candidate_records.jsonl.gz',root/'reports/VG_COLOR_FRESH_VAL_CONTROLLER_20261025.json',root/'scripts/remote_vg_color_fresh_val_execution_v1_20261025.py']},
 'test_accessed':False,'model_inference_run':False,'original_artifacts_modified':False}
print(json.dumps(result))
