"""Local source/contract audit only. No inference, metrics or heldout data."""
from pathlib import Path
from datetime import datetime,timezone
import ctypes,hashlib,importlib.util,json,time
ROOT=Path(__file__).resolve().parents[1];START=time.time()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def main():
 assert time.time()<1794670321.2414355
 kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p;h=kernel.GetCurrentProcess();mask=ctypes.c_size_t();system=ctypes.c_size_t()
 kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert kernel.GetProcessAffinityMask(h,ctypes.byref(mask),ctypes.byref(system));assert kernel.SetProcessAffinityMask(h,mask.value & -mask.value)
 manifest=ROOT/'configs/vg_grounding_controls_source_manifest_v1_20261102.json';lock=json.loads(manifest.read_text())
 for rel,h in {**lock['sha256'],**lock['local_parent_input_sha256']}.items():assert sha(ROOT/rel)==h,rel
 for rel in lock['sha256']:
  if rel.endswith('.py'):compile((ROOT/rel).read_text(),rel,'exec')
 spec=importlib.util.spec_from_file_location('frozen_new_grounding',ROOT/'src/rsgformer/vg_grounding_controls_v1_20261102.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 synthetic=module.synthetic_contract();parent=ROOT/'outputs/vg_focus_repair_failure_20261030/parent/train_generation'
 units=json.loads((parent/'query_units_by_policy.json').read_text())['L2'];assert len(units)==800
 records={g['unit_id']:g for g in map(json.loads,(parent/'generation_commit.jsonl').read_text().splitlines()) if g['policy']=='L2'};assert len(records)==800
 node_choices=edge_choices=0
 for u in units:
  assert u['split']=='train' and u['partition']=='train_bank'
  g=u['graph'];assert records[u['unit_id']]['query_graph_sha256']==digest(g)
  baseline=module.literal_first(g)
  if baseline['choice']:
   parsed=module.parse_and_render(json.dumps(baseline['choice']),g);assert parsed['caption']==baseline['caption']
  for n in g['nodes']:
   parsed=module.parse_and_render(json.dumps({'node_id':n['id']}),g);assert parsed['caption']==n['label'];node_choices+=1
  labels={n['id']:n['label'] for n in g['nodes']}
  for i,e in enumerate(g['edges']):
   parsed=module.parse_and_render(json.dumps({'edge_id':'e'+str(i)}),g)
   assert parsed['caption']==labels[e['subject']]+' '+e['predicate']+' '+labels[e['object']];edge_choices+=1
  assert module.parse_and_render('{"node_id":"not_in_graph"}',g)['caption']==''
 assert time.time()-START<75
 result={'status':'PASS_NEW_GROUNDING_CORE_AND_CPU_PREPARATION_CONTRACTS_NO_GPU_LAUNCH','utc':datetime.now(timezone.utc).isoformat(),'source_manifest_sha256':sha(manifest),'source_config_doc_files_verified':len(lock['sha256']),'parent_input_files_verified':len(lock['local_parent_input_sha256']),
  'synthetic_contract':synthetic,'saved_TRAIN_graphs_and_G0_bindings_checked':800,'valid_node_choice_roundtrips':node_choices,'valid_directed_edge_choice_roundtrips':edge_choices,
  'invalid_choice_empty_denominator_preserved':True,'query_references_read':False,'VAL_or_TEST_data_read':False,'caption_generation_or_metrics_run':False,'server_sync_or_launch':False,'main_GPU_experiment_fully_implemented':False,'wall_seconds':time.time()-START}
 (ROOT/'reports/VG_GROUNDING_CONTROLS_LOCAL_CONTRACT_20261102.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result))
if __name__=='__main__':main()
