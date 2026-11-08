"""Regression for actual clipped bbox float32 and all frozen target geometries."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,json,math,sqlite3
import numpy as np
from rsgformer.numpy_json_adapter_v1 import numpy_json_default
from rsgformer.vg_object_color_rules_v3 import clipped_box
root=Path.home()/'rsg-former';old=root/'artifacts/color_fresh_val_execution_v1_20261025'
targets=json.loads((old/'target_manifest.json').read_text())
with sqlite3.connect((root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
 dims={i:(w,h) for i,w,h in db.execute('SELECT vg_id,width,height FROM pixels')}
types=set();count=0;failed_key_seen=False
for t in targets:
 b,v=clipped_box(np.asarray(t['node']['bbox_xyxy'],dtype=np.float32),*dims[t['image_id']]);types.add(type(v).__name__)
 payload={'bbox':b,'visible_fraction':v,'decision':np.bool_(v>=.90),'pixels':np.int64(256)}
 restored=json.loads(json.dumps(payload,default=numpy_json_default,allow_nan=False))
 assert restored['visible_fraction']==float(v) and restored['decision']==bool(v>=.90)
 if t['key']=='130:g130':assert v<.90;failed_key_seen=True
 count+=1
with gzip.open(old/'candidate_records.jsonl.gz','rt') as f:prefix=list(map(json.loads,f))
assert len(prefix)==53 and failed_key_seen
for row in prefix:assert json.loads(json.dumps(row,default=numpy_json_default,allow_nan=False))==row
for v in [np.float32('nan'),np.float64('inf')]:
 try:json.dumps({'x':v},default=numpy_json_default,allow_nan=False)
 except ValueError:pass
 else:raise AssertionError('nonfinite_must_fail')
try:json.dumps({'x':object()},default=numpy_json_default)
except TypeError:pass
else:raise AssertionError('unknown_object_must_fail')
result={'status':'JSON_REPAIR_REGRESSION_PASS','utc':datetime.now(timezone.utc).isoformat(),'all_frozen_target_bbox_checks':count,'numeric_types_seen':sorted(types),
 'actual_failed_node_reproduced':True,'preserved_prefix_rows_roundtrip_equal':len(prefix),'nonfinite_rejected':True,'unsupported_objects_rejected':True,
 'model_or_mask_inference_run':False,'scientific_rules_changed':False,'TEST_accessed':False}
(root/'reports/VG_COLOR_JSON_REPAIR_CONTRACT_20261025.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
