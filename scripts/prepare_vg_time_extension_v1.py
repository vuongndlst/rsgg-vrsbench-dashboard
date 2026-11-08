"""Prepare a time-only execution revision; preserve the completed scientific run."""
from pathlib import Path
import ast,copy,hashlib,json,difflib

root=Path(__file__).absolute().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=root/'scripts/remote_vg_v2_speed_pipeline.py'
assert sha(source)=='f70681c72e2ee38aef0d869673ad58347379291c6a532254b4ef82e8ec5dc242'
old=json.loads((root/'configs/vg_method_v2_speed_m2_v1.json').read_text(encoding='utf8'))
cfg=copy.deepcopy(old)
cfg['resource']['wall_hours']=840
cfg['execution_revision']='vg_method_v2_time_extension_v1'
cfg['authorized_wall_deadline_unix']=old['original_wall_deadline_unix']+(840-240)*3600
cfg['time_extension_authorization']='Human14Oct2026: continue; time is not important; seek one best research pipeline. Time only; GPU/RAM/CPU/artifact/scientific/test gates unchanged.'
for key in old:
    if key!='resource':assert cfg[key]==old[key],key
assert {k:v for k,v in cfg['resource'].items() if k!='wall_hours'}=={k:v for k,v in old['resource'].items() if k!='wall_hours'}
config=root/'configs/vg_method_v2_time_extension_v1.json'
pipeline=root/'scripts/remote_vg_v2_time_extension_v1.py'
assert not config.exists() and not pipeline.exists(),'once_only_preparation'
text=source.read_text(encoding='utf8')
replacements=[
 ("ROOT/'configs/vg_method_v2_speed_m2_v1.json'","ROOT/'configs/vg_method_v2_time_extension_v1.json'",1),
 ("(ART/'preregistration.json')","(ART/'time_extension_v1/preregistration.json')",1),
 ("ROOT/'reports/VG_V2_SPEED_PROGRESS.json'","ROOT/'reports/VG_V2_TIME_EXT_PROGRESS.json'",1),
 ("ROOT/'reports/VG_V2_SPEED_RESULT.json'","ROOT/'reports/VG_V2_TIME_EXT_RESULT.json'",5),
 ("CFG['original_wall_deadline_unix']","CFG['authorized_wall_deadline_unix']",2),
]
# Keep ORIGINAL_START anchored to the original accounting, rather than resetting time.
text=text.replace("ORIGINAL_START=CFG['original_wall_deadline_unix']-240*3600", "ORIGINAL_START=1792510321.2414355-240*3600")
for a,b,n in replacements:
    assert text.count(a)==n,(a,text.count(a),n)
    text=text.replace(a,b)
text=text.replace("raise RuntimeError('original_absolute_wall_budget_reached')", "raise RuntimeError('authorized_extended_wall_budget_reached')")
text=text.replace("registered confirmation and test allowance do not fit original absolute240h cap; no gate weakening or partial confirmation", "registered confirmation and test allowance do not fit authorized extended deadline; no gate weakening or partial confirmation")
text=text.replace(" atomic(ART/'frozen_native_validation.json',frozen)\n shutil.copy2(ROOT/'artifacts/vg_method_v2/frozen_native_validation_records.json',ART/'frozen_native_validation_records.json')", " # Read existing frozen baseline; preserve original completed evidence files.")
text=text.replace("atomic(ART/'initialization.json'", "atomic(ART/'time_extension_v1/initialization.json'")
text=text.replace(" verify();guard();event('resource_admission')", " verify();guard()\n for name,h in PREREG['inherited_screening_file_hashes'].items():assert sha(ROOT/name)==h,'inherited_screening_evidence_changed'\n event('resource_admission',execution_revision=CFG['execution_revision'])")
ast.parse(text)
def function_ast(s,name):
    tree=ast.parse(s)
    return ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name),include_attributes=False)
for name in ['run','items','evaluate','feature_cache','bootstrap','resources','artifact_guard']:
    assert function_ast(text,name)==function_ast(source.read_text(encoding='utf8'),name),name
assert text.index("if not valid:event('sgdet_stage_finished'")<text.index("test=datasets(ROOT,'test')")<text.index("atomic(ART/'test_lock.json'")<text.index("feature_cache(test,'test')")
assert "range(1,epochs+1)" in text and "(values.sum()/16).backward()" in text
config.write_text(json.dumps(cfg,indent=2)+'\n',encoding='utf8')
pipeline.write_text(text,encoding='utf8')
diff=''.join(difflib.unified_diff(source.read_text(encoding='utf8').splitlines(True),text.splitlines(True),fromfile=source.name,tofile=pipeline.name))
(root/'research/VG_TIME_EXTENSION_V1_SOURCE_DIFF.txt').write_text(diff,encoding='utf8')
fixture={'status':'PASS','original_pipeline_sha256':sha(source),'new_pipeline_sha256':sha(pipeline),'new_config_sha256':sha(config),'unchanged_function_ast':['run','items','evaluate','feature_cache','bootstrap','resources','artifact_guard'],'original_accounting_preserved':True,'screening_will_be_reused_after_file_hash_checks':True,'test_gate_order_preserved':True,'resource_change_only':'wall_hours240->840; absolute original accounting retained','training_launched':False,'test_scored':False}
(root/'reports/VG_TIME_EXTENSION_V1_FIXTURE.json').write_text(json.dumps(fixture,indent=2),encoding='utf8')
print(json.dumps(fixture))
