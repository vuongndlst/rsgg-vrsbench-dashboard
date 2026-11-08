"""One-pass color-only cache on frozen TEST predictions; no GT or SGDet scoring."""
from pathlib import Path
import collections,gc,gzip,hashlib,json,os,resource,signal,sqlite3,time,traceback
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUDA_CACHE_DISABLE='1')
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_color_full_test_auto_v1_20261025';PIPE=ROOT/'artifacts/vg_color_auto_full_pipeline_v1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(82800)
from rsgformer.vg_color_execution_common_v1 import Guard,ImageZIPs,load_models,sha,save,line,now,artifact_bytes
from rsgformer.vg_color_full_test_policy_v1 import select_nodes,color_sidecar
from rsgformer.vg_caption_graph_v1 import compact_graph

def main():
 assert not ART.exists(),'no_duplicate_or_partial_retry'
 guard=Guard(ROOT,ART,82800,6);guard.check(False)
 lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h,('source_drift',p)
 cfg=json.loads((ROOT/'configs/vg_color_automated_full_test_v1_20261025.json').read_text())
 assert sha(ROOT/'configs/vg_color_automated_full_test_v1_20261025.json')==lock['config_sha256']
 parity=json.loads((ROOT/'artifacts/vg_color_fast_val_replay_v1_20261025/complete.json').read_text())
 assert parity['status']=='FAST_VAL_REPLAY_PARITY_PASS' and parity['nodes']==3083
 auto=json.loads((ROOT/'artifacts/vg_color_auto_val_diagnostic_v1_20261025/complete.json').read_text())
 assert auto['counts']['selected_nodes']==3083 and auto['human_precision_gate_passed'] is False
 source=ROOT/'artifacts/vg_component_control_test_v1/test_predictions.jsonl.gz'
 assert sha(source)==cfg['TEST_prediction_sha256']
 checkpoint=ROOT/'artifacts/vg_method_v2_speed_m2_v1/confirmation_context_1729/best.pt'
 assert sha(checkpoint)==cfg['checkpoint_sha256']
 import shutil,subprocess
 assert shutil.disk_usage(ROOT).free>=22*1024**3,'reserve_plus_full_6GiB_admission'
 assert artifact_bytes(ROOT/'artifacts')+6*1024**3<=40*1024**3,'aggregate_worst_case_admission'
 assert int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])>=8*1024
 ART.mkdir()
 save(ART/'preregistration.json',{'utc':now(),'pipeline_preregistration_sha256':sha(PIPE/'preregistration.json'),'config_sha256':lock['config_sha256'],
  'source_sha256':lock['source_sha256'],'VAL_parity_sha256':sha(ROOT/'artifacts/vg_color_fast_val_replay_v1_20261025/complete.json'),
  'TEST_prediction_sha256':cfg['TEST_prediction_sha256'],'human_precision_gate_passed':False,'status':'exploratory_candidate_only','no_GT_in_extractor':True})
 engine,torch=load_models(ROOT,guard)
 # Membership and archive directory entries are metadata, not TEST attributes or semantic model inputs.
 with sqlite3.connect((ROOT/'artifacts/caption_readiness/vg_pixel_fingerprints_v1.sqlite').as_uri()+'?mode=ro',uri=True) as db:
  expected={int(r[0]) for r in db.execute("SELECT vg_id FROM pixels WHERE split='test'")}
 assert len(expected)==26446
 images=ImageZIPs(ROOT,expected)
 vocabulary=json.loads((ROOT/'data/vg150/VG-SGG-dicts-with-attri.json').read_text())
 objects={int(k):v for k,v in vocabulary['idx_to_label'].items()};predicates={int(k):v for k,v in vocabulary['idx_to_predicate'].items()}
 eligible=set(cfg['eligible_classes']);seen=set();counts=collections.Counter();start=time.monotonic()
 ledger={'utc':now(),'status':'OPENED_EXCLUSIVE_COLOR_ACCESS_NO_RETRY','pid':os.getpid(),'expected_images':26446,
  'source_predictions_sha256':cfg['TEST_prediction_sha256'],'GT_access_allowed':False,'new_detector_or_SGDet_scoring':False}
 with (ART/'test_access.json').open('x') as f:f.write(line(ledger)+'\n');f.flush();os.fsync(f.fileno())
 with gzip.open(source,'rt') as inp,gzip.open(ART/'candidate_records.jsonl.gz','xt',compresslevel=3) as out,gzip.open(ART/'full_sg_color_candidates.jsonl.gz','xt',compresslevel=3) as sgout,(ART/'image_commits.jsonl').open('x') as commits:
  for source_index,text in enumerate(inp):
   guard.check();raw=json.loads(text);identity=int(raw['image_id']);assert identity in expected and identity not in seen
   graph=compact_graph(raw['prediction'],objects,predicates);selected=select_nodes(graph,eligible,cfg['maximum_requested_nodes_per_image']);records={}
   if selected:
    engine.set_image(images.read(identity),identity);counts['images_with_requested_nodes']+=1
    # Selection priority controls membership. Within-image processing order matches the VAL replay.
    for node in sorted(selected,key=lambda n:str(identity)+':'+n['id']):
     guard.check();key=str(identity)+':'+node['id'];target={'key':key,'image_id':identity,'partition':'test','node':node}
     row=engine.run(target,graph);row.update(image_id=identity,partition='test',node=node,**{'class':node['label']})
     out.write(line(row)+'\n');records[node['id']]=row;counts['requested_nodes']+=1
     if counts['requested_nodes']%16==0:gc.collect()
   else:counts['images_without_requested_nodes']+=1
   annotations=color_sidecar(graph,selected,records,eligible)
   for annotation in annotations.values():counts['node_'+annotation['status']]+=1
   item={'image_id':identity,'source_index':source_index,'graph':graph,'color_annotations':annotations,
    'colors_are_unverified_candidates':True,'human_precision_gate_passed':False,'prediction_source_sha256':cfg['TEST_prediction_sha256']}
   serialized=line(item);sgout.write(serialized+'\n')
   commits.write(line({'image_id':identity,'source_index':source_index,'requested_nodes':len(selected),'SG_row_sha256':hashlib.sha256(serialized.encode()).hexdigest()})+'\n')
   seen.add(identity);counts['images']+=1
   if counts['images']%100==0:
    out.flush();sgout.flush();commits.flush();save(ART/'progress.json',{'utc':now(),'counts':dict(counts),'expected_images':26446,'TEST_accessed':True,'human_precision_gate_passed':False,'elapsed_seconds':time.monotonic()-start})
 images.close();assert seen==expected and counts['images']==26446,'full_denominator_required'
 assert sha(source)==cfg['TEST_prediction_sha256'],'immutable_input_drift'
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 guard.check()
 outputs={name:sha(ART/name) for name in ['candidate_records.jsonl.gz','full_sg_color_candidates.jsonl.gz','image_commits.jsonl']}
 result={'status':'FULL_COLOR_TEST_CACHE_COMPLETE_UNVERIFIED_CANDIDATES','utc':now(),'counts':dict(counts),'expected_images':26446,'one_pass':True,
  'TEST_prediction_sha256':cfg['TEST_prediction_sha256'],'output_sha256':outputs,'human_labels':0,'human_precision_gate_passed':False,
  'elapsed_seconds_after_load':time.monotonic()-start,'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,'peak_allocator_bytes':torch.cuda.max_memory_allocated(),
  'GT_accessed_by_extractor':False,'new_SGDet_inference_or_scoring':False,'caption_generated':False}
 save(ART/'complete.json',result);save(ROOT/'reports/VG_COLOR_FULL_TEST_AUTO_V1_20261025.json',result);print(line(result),flush=True)

if __name__=='__main__':
 try:main()
 except Exception as error:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'error_type':type(error).__name__,'reason':str(error),'TEST_accessed':(ART/'test_access.json').exists(),'automatic_retry_allowed':False})
  traceback.print_exc();raise
