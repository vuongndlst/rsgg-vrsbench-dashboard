"""User19Oct amendment: validation-locked control, validation gates, one TEST pass."""
from pathlib import Path
from datetime import datetime,timezone
import ast,gzip,hashlib,io,json,os,random,resource,shutil,subprocess,sys,textwrap,time,traceback
ROOT=Path.home()/'rsg-former';ART=ROOT/'artifacts/vg_component_control_test_v1'
LOCK=json.loads((ART/'preregistration.json').read_text());START=time.time()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8388608),b''):h.update(b)
 return h.hexdigest()
def atomic(p,x):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2));t.replace(p)
def verify():
 for p,h in LOCK['source_hashes'].items():assert sha(ROOT/p)==h,p
 assert sha(ROOT/LOCK['checkpoint'])==LOCK['checkpoint_sha256']
def guard():
 assert not (ART/'STOP').exists(),'owned_component_stop'
 assert time.time()<LOCK['deadline_unix'],'component_wall_cap'
 assert shutil.disk_usage(ROOT).free>=16*1024**3,'disk_reserve'
 assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=4*1024**3,'rss_cap'
def event(stage,**kw):
 guard();atomic(ROOT/'reports/VG_COMPONENT_CONTROL_TEST_PROGRESS.json',{'utc':datetime.now(timezone.utc).isoformat(),'stage':stage,'elapsed_seconds':time.time()-START,**kw})
def artifacts():
 own=sum(p.stat().st_size for p in ART.rglob('*') if p.is_file());assert own<=2*1024**3,'component_artifact_cap'
 total=own+sum(p.stat().st_size for base in ['vg_method_v2','vg_method_v2_speed_m2_v1'] for p in (ROOT/'artifacts'/base).rglob('*') if p.is_file())
 assert total<=40*1024**3,'aggregate_artifact_cap'
def resources():
 guard()
 p=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);assert p.returncode==0
 own=[int(s.split(',')[1])*1024**2 for s in p.stdout.splitlines() if int(s.split(',')[0])==os.getpid()]
 assert len(own)==1 and own[0]<=8*1024**3,'own_gpu_cap'
 return {'own_gpu_bytes':own[0],'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024}
def main():
 verify();guard();os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]))
 import numpy as np,torch,torchvision
 from rsgformer.vg_v2_runtime import native_model,datasets,capture,Adaptation,SGDetEvaluator,predicted_graph
 from rsgformer.vg_canonical_data import VGCanonical
 from rsgformer.vg_canonical_evaluation import CanonicalSGDetEvaluator,inclusive_iou_float64
 from rsgformer.vg_native_postprocess import native_vg_postprocess
 torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.set_float32_matmul_precision('highest');torch.manual_seed(1729);random.seed(1729);np.random.seed(1729)
 torch.cuda.set_per_process_memory_fraction(6*1024**3/torch.cuda.get_device_properties(0).total_memory)
 model,_,_=native_model(ROOT);ck=torch.load(ROOT/LOCK['checkpoint'],map_location='cpu',weights_only=True)
 assert ck['variant']=='context' and ck['seed']==1729 and ck['epoch']==10 and ck['microbatch']==2
 assert set(ck['native_head'])=={k for k in model.state_dict() if 'add_add_add_relation' in k},'checkpoint_head_key_mismatch'
 loaded=model.load_state_dict(ck['native_head'],strict=False);assert not loaded.unexpected_keys
 adapt=Adaptation(model,'context');adapt.residual.load_state_dict(ck['residual'],strict=True)
 for p in adapt.parameters():p.requires_grad_(False)
 adapt.cuda().eval();model.eval();del ck;resources()
 val=datasets(ROOT,'val');canon=VGCanonical(ROOT/'data/vg_canonical_audit/reltr_vg_annotations.zip',Path.home()/'sbir/hienlth/vg','val')
 canonical_index={x:i for i,x in enumerate(canon.ids)};assert set(val.split_image_ids())==set(canon.ids) and len(val)==5000
 # Exercise the full frozen detector on a validation image before opening TEST.
 with torch.no_grad():
  image=canon.image(0)
  try:smoke_item,smoke_out=capture(model,image)
  finally:image.close()
  assert all(torch.isfinite(x).all() for x in smoke_item.values()),'validation_detector_nonfinite'
  resources();del smoke_item,smoke_out;torch.cuda.empty_cache()
 legacy=SGDetEvaluator();official=CanonicalSGDetEvaluator(ROOT);reference_values={k:[] for k in (20,50,100)};reference_predicates={k:[[] for _ in range(50)] for k in (20,50,100)}
 # Extract the actual pinned upstream postprocessing block, without touching vendor source.
 source=(ROOT/'vendor/salience_sgg/models/salience_sggmodule.py').read_text()
 begin=source.index('            pred_rel = torch.clamp(');end=source.index("            if 'oi_sgg_evaluator'",begin);first=textwrap.dedent(source[begin:end])
 begin=source.index('            demo_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)');end=source.index('            connections =',begin);second=textwrap.dedent(source[begin:end])
 function="def reference(out):\n    j=0\n    pred_boxes=out['pred_boxes'][0]\n    pred_logits=out['pred_logits'][0].sigmoid()\n    obj_scores,pred_classes=pred_logits.softmax(-1).max(-1)\n    pred_rel=out['pred_rel'][0].sigmoid()\n    rel_confideces=out['pred_confidence'][-1].sigmoid()\n"+textwrap.indent(first+second,'    ')+"\n    return {'pred_rel_inds':demo[:100].cpu().numpy(),'rel_scores':result_pred_rel_scores[:100].cpu().numpy(),'ranking_scores':triplet_scores[sorting_idx][keep_triplet==1][:100].cpu().numpy(),'query_to_nms_equivalence_id':iou_assignments.cpu().numpy()}\n"
 from util import box_ops
 ns={'torch':torch,'torchvision':torchvision,'box_ops':box_ops};exec(compile(ast.parse(function),'<pinned_upstream_validation_postprocess>','exec'),ns)
 def independent(sample,pred):
  gt=sample.relations;boxes=pred['pred_boxes'];classes=pred['pred_classes']+1;pairs=pred['pred_rel_inds'];labels=pred['rel_scores'].argmax(1)+1;iou=inclusive_iou_float64(sample.boxes,boxes);hits=[]
  for (s,o),label in zip(pairs,labels):hits.append(set(np.where((gt[:,2]==label)&(sample.labels[gt[:,0]]==classes[s])&(sample.labels[gt[:,1]]==classes[o])&(iou[gt[:,0],s]>=.5)&(iou[gt[:,1],o]>=.5))[0]))
  for k in reference_values:
   matched=set().union(*hits[:k]) if hits else set();reference_values[k].append(len(matched)/len(gt))
   for p in np.unique(gt[:,2]):inds=set(np.where(gt[:,2]==p)[0]);reference_predicates[k][p-1].append(len(inds&matched)/len(inds))
 def batch(item):return {k:item[k][None].float().cuda() for k in ['nodes','boxes','logits','visual']}
 manifest=ROOT/'artifacts/vg_method_v2/features/manifest.json';assert sha(manifest)==LOCK['cache_manifest_sha256'];cache=json.loads(manifest.read_text());seen=[];n=0;parity_images=0
 with torch.no_grad(),gzip.open(ART/'validation_predictions.jsonl.gz','xt',encoding='utf8') as stream:
  for entry in cache['splits']['val']:
   p=manifest.parent/entry['path'];assert sha(p)==entry['sha256'];chunk=torch.load(p,map_location='cpu',weights_only=True);assert [x['image_id'] for x in chunk]==entry['image_ids']
   for item in chunk:
    guard();sample=val[item['index']];assert sample.image_id==item['image_id'];canonical=canon[canonical_index[sample.image_id]];out,_=adapt(batch(item));legacy.update(sample,*predicted_graph(out,sample));pred=native_vg_postprocess(out,sample.width,sample.height)
    if n<32:
     expected=ns['reference'](out)
     for key in expected:
      if key in ('pred_rel_inds','query_to_nms_equivalence_id'):assert np.array_equal(pred[key],expected[key]),'real_postprocessor_pair_parity:'+key
      else:assert np.allclose(pred[key],expected[key],rtol=1e-6,atol=1e-8),'real_postprocessor_numeric_parity:'+key
     parity_images+=1
    independent(canonical,pred);official.update(canonical,pred);stream.write(json.dumps({'image_id':sample.image_id,'prediction':{k:v.tolist() if hasattr(v,'tolist') else v for k,v in pred.items()}})+'\n');seen.append(sample.image_id);n+=1
    if n%128==0:stream.flush();artifacts();event('validation_pretest_gate',images=n,target_images=5000,**resources())
 assert len(seen)==5000 and len(set(seen))==5000
 lm=legacy.result();cm=official.result()
 for k in (20,50,100):
  assert abs(lm[f'R@{k}']-LOCK['selection_validation_metrics'][f'R@{k}'])<1e-8,'selected_checkpoint_validation_not_reproduced'
  assert abs(lm[f'mR@{k}']-LOCK['selection_validation_metrics'][f'mR@{k}'])<1e-8,'selected_checkpoint_validation_not_reproduced'
  assert abs(cm[f'R@{k}']-float(np.mean(reference_values[k])))<1e-12,'real_evaluator_recall_parity'
  mr=sum(float(np.mean(x)) if x else 0 for x in reference_predicates[k])/50;assert abs(cm[f'mR@{k}']-mr)<1e-12,'real_evaluator_macro_parity'
 validation={'status':'PASS','images':5000,'full_detector_validation_smoke':'PASS','postprocess_source_parity_images':parity_images,'legacy_selection_reproduced':lm,'canonical_metrics_before_test':cm,'independent_float64_evaluator_parity':'PASS','checkpoint_frozen_before_this_validation':True,'test_scored':False}
 atomic(ART/'validation_gate.json',validation);atomic(ROOT/'reports/VG_COMPONENT_CONTROL_VALIDATION_GATE.json',validation)
 # No selection/calibration changes after validation; TEST is opened exactly once.
 verify();guard();assert not (ART/'test_access.json').exists();assert not (ROOT/'reports/VG_COMPONENT_TEST_ACCESS_LEDGER.json').exists()
 access={'utc':datetime.now(timezone.utc).isoformat(),'status':'TEST_OPENED_ONCE','preregistration_sha256':sha(ART/'preregistration.json'),'checkpoint_sha256':LOCK['checkpoint_sha256'],'selection_frozen':True,'future_tuning_from_test_prohibited':True}
 with (ART/'test_access.json').open('x') as f:json.dump(access,f)
 with (ROOT/'reports/VG_COMPONENT_TEST_ACCESS_LEDGER.json').open('x') as f:json.dump(access,f)
 test=VGCanonical(ROOT/'data/vg_canonical_audit/reltr_vg_annotations.zip',Path.home()/'sbir/hienlth/vg','test');assert len(test)==26446
 ev=CanonicalSGDetEvaluator(ROOT);images=0
 with torch.no_grad(),gzip.open(ART/'test_predictions.jsonl.gz','xt',encoding='utf8') as stream:
  for i in range(len(test)):
   guard();sample=test[i];image=test.image(i)
   try:item,_=capture(model,image)
   finally:image.close()
   out,_=adapt(batch(item));pred=native_vg_postprocess(out,sample.width,sample.height);ev.update(sample,pred);stream.write(json.dumps({'image_id':sample.image_id,'prediction':{k:v.tolist() if hasattr(v,'tolist') else v for k,v in pred.items()}})+'\n');images+=1
   if images%128==0:stream.flush();artifacts();event('locked_component_test',images=images,target_images=len(test),**resources())
 assert images==26446 and ev.result()['observed_images']==26446
 result={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPONENT_TEST_COMPLETE','model':'Salience-SGG R50 + context control','variant':'context','seed':1729,'selected_epoch':10,'checkpoint_sha256':LOCK['checkpoint_sha256'],'validation_gate':validation,'test':ev.result(),'test_scored':True,'test_passes':1,'strict_native_training_recipe_reproduction':False,'canonical_targets_and_pinned_evaluator':True,'input_preprocessing':'unchanged component FP32 cached-training resize(round)800/1333, no test calibration','comparability_limitations':LOCK['comparability_limitations'],'clean_release_certified':False,'caption_evaluated':False,'wall_seconds':time.time()-START,'artifact_hashes':{p.name:sha(p) for p in ART.glob('*.jsonl.gz')}}
 atomic(ART/'result.json',result);atomic(ROOT/'reports/VG_COMPONENT_CONTROL_TEST_RESULT.json',result);event('component_test_complete',images=images,test_scored=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  result={'utc':datetime.now(timezone.utc).isoformat(),'status':'COMPONENT_TEST_BLOCKED','error_type':type(e).__name__,'reason':str(e)[:1000],'traceback':traceback.format_exc()[-4000:],'test_opened':(ART/'test_access.json').exists(),'test_scored':False,'no_blind_retry':True}
  atomic(ROOT/'reports/VG_COMPONENT_CONTROL_TEST_RESULT.json',result);raise
