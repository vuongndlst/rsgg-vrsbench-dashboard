"""Read-only layout of VAL predictions and train cache; no TEST rows or model."""
from pathlib import Path
from datetime import datetime,timezone
import gzip,json,os,sqlite3,zipfile
root=Path.home()/'rsg-former';os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
pred=root/'artifacts/vg_component_control_test_v1/validation_predictions.jsonl.gz'
with gzip.open(pred,'rt') as f:first=json.loads(next(f))
def shape(x):
 return [len(x)]+shape(x[0]) if isinstance(x,list) and x else []
with zipfile.ZipFile(root/'data/vg_canonical_audit/reltr_vg_annotations.zip') as z:
 meta=json.loads(z.read('vg/val.json'));relation=json.loads(z.read('vg/rel.json'))
mf=json.loads((root/'artifacts/vg_method_v2/features/manifest.json').read_text())
with sqlite3.connect((root/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
 schema=db.execute('PRAGMA table_info(pixels)').fetchall();row=db.execute('SELECT * FROM pixels LIMIT 1').fetchone()
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'status':'LAYOUT_PROBE_COMPLETE','test_rows_read':False,
 'val_prediction_bytes':pred.stat().st_size,'prediction_shapes':{k:shape(v) for k,v in first['prediction'].items()},
 'first_prediction_preview':{k:v[:2] if isinstance(v,list) else v for k,v in first['prediction'].items() if k!='rel_scores'},
 'first_val_image_metadata':meta['images'][0],'categories':meta['categories'],'predicate_names':relation['rel_categories'],
 'cache_first_train_entry':mf['splits']['train'][0],'cache_first_val_entry':mf['splits']['val'][0],
 'pixel_schema':schema,'pixel_first_row':row,'prediction_names':list(first['prediction'])}))
