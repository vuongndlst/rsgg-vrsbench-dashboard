"""All3083VAL-record engineering parity audit before any full colorTEST access."""
from pathlib import Path
import gc,gzip,json,os,resource,signal,time,traceback
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1')
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_fast_val_replay_v1_20261025';PARENT=ROOT/'artifacts/color_fresh_val_execution_r1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(3600)
from rsgformer.vg_color_execution_common_v1 import Guard,ImageZIPs,load_models,sha,save,line,now,difference
from rsgformer.vg_color_full_test_policy_v1 import select_nodes
def main():
 assert not ART.exists();guard=Guard(ROOT,ART,3600,.5);guard.check(False);ART.mkdir()
 complete=json.loads((PARENT/'complete.json').read_text());assert complete['nodes']==3083 and sha(PARENT/'candidate_records.jsonl.gz')==complete['candidate_records_sha256']
 target_list=json.loads((PARENT/'target_manifest.json').read_text());targets={r['key']:r for r in target_list};ids={r['image_id'] for r in target_list}
 sources=['scripts/remote_vg_color_fast_val_replay_v1_20261025.py','src/rsgformer/vg_color_execution_common_v1.py','src/rsgformer/vg_color_fast_engine_v1.py','src/rsgformer/vg_object_color_rules_v3.py','src/rsgformer/vg_object_color_v2.py','src/rsgformer/vg_color_binding_native_v1.py','src/rsgformer/numpy_json_adapter_v1.py']
 save(ART/'preregistration.json',{'utc':now(),'source_sha256':{p:sha(ROOT/p) for p in sources},'reference_records_sha256':complete['candidate_records_sha256'],
  'expected_nodes':3083,'tolerance':1e-8,'all_categorical_decisions_must_match':True,'no_GT_given_to_extractor':True,'TEST_access_allowed':False,'no_threshold_tuning':True})
 with gzip.open(ROOT/'artifacts/vg_caption_execution_v1_20261025/val_base_sg.jsonl.gz','rt') as f:graphs={r['image_id']:r['graph'] for r in map(json.loads,f) if r['image_id'] in ids}
 cfg=json.loads((ROOT/'configs/vg_color_automated_full_test_v1_20261025.json').read_text())
 expected_by_image={i:{t['node']['id'] for t in target_list if t['image_id']==i} for i in ids}
 assert all({n['id'] for n in select_nodes(g,set(cfg['eligible_classes']),cfg['maximum_requested_nodes_per_image'])}==expected_by_image[i] for i,g in graphs.items()),'VAL_selection_contract_drift'
 images=ImageZIPs(ROOT,ids);engine,torch=load_models(ROOT,guard);current=None;count=0;maximum=0.;disagreements=[];start=time.monotonic()
 with gzip.open(PARENT/'candidate_records.jsonl.gz','rt') as f,gzip.open(ART/'optimized_records.jsonl.gz','xt') as out:
  for old in map(json.loads,f):
   guard.check();target=targets[old['key']];identity=old['image_id']
   if identity!=current:engine.set_image(images.read(identity),identity);current=identity
   new=json.loads(line(engine.run(target,graphs[identity])))
   for field in ['methods','CLIP_fill127_fill255','native_object_binding','excluded_child_node_ids']:
    paths,diff=difference(old[field],new[field]);maximum=max(maximum,diff)
    if paths:disagreements.append({'key':old['key'],'field':field,'paths':paths[:8]})
   out.write(line(new)+'\n');count+=1
   if count%16==0:gc.collect()
   if count%100==0:out.flush();save(ART/'progress.json',{'utc':now(),'nodes':count,'expected':3083,'difference_cases':len(disagreements),'TEST_accessed':False})
 images.close();assert count==3083
 result={'status':'FAST_VAL_REPLAY_PARITY_PASS' if not disagreements else 'FAST_VAL_REPLAY_PARITY_FAIL_TEST_BLOCKED','utc':now(),'nodes':count,'difference_cases':len(disagreements),
  'maximum_numeric_difference':maximum,'differences':disagreements[:30],'masks_note':'direct consumed-mask hashes were compared on fixedTRAIN benchmark; fullVAL compares all saved scientific outputs/diagnostics, original fullVAL mask hashes were not stored',
  'seconds_after_model_load':time.monotonic()-start,'peak_allocator_bytes':torch.cuda.max_memory_allocated(),'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
  'optimized_records_sha256':sha(ART/'optimized_records.jsonl.gz'),'TEST_accessed':False,'human_precision_gate_passed':False}
 save(ART/'complete.json',result);save(ROOT/'reports/VG_COLOR_FAST_VAL_REPLAY_V1_20261025.json',result);print(line(result),flush=True)
 assert not disagreements,'engineering_parity_failed_before_TEST'
if __name__=='__main__':
 try:main()
 except Exception as error:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'error_type':type(error).__name__,'reason':str(error),'TEST_accessed':False})
  traceback.print_exc();raise
