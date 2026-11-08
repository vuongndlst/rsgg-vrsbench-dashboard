"""Invented annotation contracts; no dataset, pixels, model or scoring access."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, importlib.util, json
root=Path(__file__).absolute().parents[1]
source=root/'src/rsgformer/vg_caption_region_units.py'
spec=importlib.util.spec_from_file_location('crop_contract',source)
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
base={'region_id':1,'image_id':7,'x':0,'y':0,'width':10,'height':10,'phrase':'invented human reference'}
rows=[base,dict(base,region_id=2,x=1.2,y=2.1,width=3.2,height=4.1),
      dict(base,region_id=3),dict(base,region_id=4,x=-1),
      dict(base,region_id=5,width=0),dict(base,region_id=6,x=9,width=2),
      dict(base,region_id=7,x=float('nan')),dict(base,region_id=8,phrase='  '),
      dict(base,region_id=9,x=True),dict(base,region_id=10,width=1e309)]
inputs,refs,rejected=module.prepare_provided_crop_units(7,(10,10),rows)
assert len(inputs)==3 and len(rejected)==7 and len(inputs)+len(rejected)==len(rows)
assert inputs['vg:7:region:1']['crop_xyxy']==[0,0,10,10]
assert inputs['vg:7:region:2']['crop_xyxy']==[1,2,5,7]
assert refs['vg:7:region:1']['human_references']==[base['phrase']]
assert refs['vg:7:region:3']['human_references']==[base['phrase']]
assert all(set(row)=={'unit_id','parent_image_id','region_id','image_size','crop_xyxy','scope'} for row in inputs.values())
assert not any('phrase' in row or 'human_references' in row for row in inputs.values())
shuffled,shuffled_refs,_=module.prepare_provided_crop_units(7,(10,10),list(reversed(rows)))
assert shuffled==inputs and shuffled_refs==refs
for altered,reason in [([base,dict(base,phrase='')],'duplicate_region_identity'),
                       ([dict(base,image_id=8)],'region_parent_identity_mismatch'),
                       ([dict(base,region_id=True)],'invalid_region_or_image_identity')]:
    try: module.prepare_provided_crop_units(7,(10,10),altered)
    except ValueError as error: assert str(error)==reason
    else: raise AssertionError('identity_integrity_guard_failed')
cfg=root/'configs/vg_region_caption_geometry_v1.json'
protocol=json.loads(cfg.read_text(encoding='utf8'))
assert not protocol['test_scores_used_for_rules'] and not protocol['live_v2_modified']
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'protocol_sha256':hashlib.sha256(cfg.read_bytes()).hexdigest(),
        'cases':['full image half-open boundary','fractional enclosing crop','duplicate geometry retains region IDs',
                 'invalid geometry and reference counts conserved','reference text absent from model input',
                 'order independence','duplicate ID fatal even ineligible','wrong parent fatal','boolean identity fatal'],
        'invented_annotation_rows':len(rows),'real_reference_text_read':False,
        'image_pixels_read_or_written':False,'network_bytes':0,'model_or_metric_run':False,
        'benchmark_test_scored':False,'live_v2_modified':False,
        'real_schema_and_geometry_audit_complete':False,'caption_evaluation_admitted':False}
(root/'reports/VG_REGION_CAPTION_GEOMETRY_PROBE.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
