"""Offline verification of immutable finished grounding TRAIN/dev evidence. No reruns."""
from pathlib import Path
from datetime import datetime,timezone
import collections,csv,gzip,hashlib,json,sys,zipfile
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from rsgformer.vg_caption_graph_v1 import digest
from rsgformer.vg_grounding_controls_v1_20261102 import literal_first,parse_and_render

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf8'))
def main():
 status=read(ROOT/'reports/VG_GROUNDING_TRAIN_DEV_R1_LAST_STATUS_20261103.json');complete=status['complete.json'];launch=read(ROOT/'reports/VG_GROUNDING_TRAIN_DEV_R1_LAUNCH_20261103.json')
 assert complete['status']=='GROUNDING_FINITE_TRAIN_DEV_COMPLETE_NO_HELDOUT_OR_TEST'
 archive=ROOT/complete['archive'];assert sha(archive)==complete['archive_sha256']=='767c7e0c0f26c67aba6c8fe9b6c8d01c44754ca92af236a90d41c023227aa157'
 out=ROOT/'outputs/vg_grounding_train_dev_r1_20261103';out.mkdir(exist_ok=False)
 with zipfile.ZipFile(archive) as z:
  for name in z.namelist():
   rel=Path(name);assert not rel.is_absolute() and '..' not in rel.parts
   dest=out/rel;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(z.read(name))
 assert sha(out/'preregistration.json')==launch['preregistration_sha256']==complete['preregistration_sha256']
 assert sha(out/'source_manifest.json')==launch['source_manifest_sha256']==complete['source_manifest_sha256']
 assert sha(out/'source_manifest.json')==sha(ROOT/'configs/vg_grounding_train_dev_source_manifest_r1_20261103.json')
 manifest=read(out/'source_manifest.json');config=read(out/'config.json');lock=read(out/'preregistration.json')
 assert sha(out/'config.json')==lock['config_sha256']==sha(ROOT/'configs/vg_grounding_train_dev_r1_20261103.json')
 for rel,h in manifest['source_sha256'].items():assert sha(ROOT/manifest['remote_metadata_local_raw_receipts'].get(rel,rel))==h,rel
 parent=ROOT/'outputs/vg_focus_repair_failure_20261030/parent';cache=ROOT/'outputs/vg_grounding_dev_prepare_v1_20261102'
 for base,expected in [(parent,config['parent_files_sha256']),(cache,config['dev_cache_files_sha256'])]:
  for rel,h in expected.items():assert sha(base/rel)==h,rel
 assert lock['parent_input_sha256']==config['parent_files_sha256'] and lock['dev_cache_input_sha256']==config['dev_cache_files_sha256']
 pre=read(out/'preflight/complete.json');assert sha(out/'preflight/complete.json')==launch['preflight_completion_sha256']==lock['preflight_completion_sha256']
 for name,h in pre['file_sha256'].items():assert sha(out/'preflight'/name)==h,name
 assert pre['G0_TRAIN_exact_reuse_matches']==800 and pre['TRAIN_units']==800 and pre['dev_units']==256 and max(pre['maximum_native_input_tokens'].values())<=768
 original_rows=[json.loads(l) for l in (parent/'train_generation/generation_commit.jsonl').read_text().splitlines()];baseline={r['unit_id']:r for r in original_rows if r['policy']=='L2'}
 original_queries={u['unit_id']:u for u in read(parent/'train_generation/query_units_by_policy.json')['L2']};assert len(baseline)==len(original_queries)==800
 reservation=read(ROOT/'configs/vg_grounding_fresh_val_reservation_v2_20261101.json');devparents={r['image_id'] for r in reservation['dev']};checkparents={r['image_id'] for r in reservation['check']}
 scores={};diagnostics={};selection_stats={};all_rows={};corpus_checks=[];new_calls=0;empty_counts={};max_gen_tokens=0
 old_metric=read(parent/'train_metrics/L2.json')
 for split,expected in [('train',800),('dev',256)]:
  gpath=out/(split+'_generation');mpath=out/(split+'_metrics');gc=read(gpath/'complete.json');mc=read(mpath/'complete.json')
  assert sha(gpath/'generation_commit.jsonl')==gc['generation_sha256']==mc['generation_sha256']
  assert sha(gpath/'queries.json')==gc['query_sha256']==sha(out/'preflight'/('queries_'+split+'.json'))
  qs={u['unit_id']:u for u in read(gpath/'queries.json')};rows=[json.loads(l) for l in (gpath/'generation_commit.jsonl').read_text().splitlines()]
  assert len(qs)==expected and len(rows)==expected*3 and len({(r['arm'],r['unit_id']) for r in rows})==expected*3
  assert gc['arms']==mc['arms']==['G0','G1','G2'] and gc['units_per_arm']==mc['units_per_arm']==expected
  refs=read(mpath/'original_human_references.json');reference_source=read(parent/'train_metrics/original_human_references.json') if split=='train' else read(cache/'dev_references_scoring_only.json')
  assert refs==reference_source and set(refs)==set(qs) and all(isinstance(v,list) and v for v in refs.values())
  if split=='train':
   assert set(qs)==set(baseline)==set(original_queries)
   for uid,q in qs.items():assert q==original_queries[uid]
  else:
   assert {u['parent_image_id'] for u in qs.values()}==devparents and not devparents&checkparents
   assert collections.Counter(u['parent_image_id'] for u in qs.values())==collections.Counter({i:4 for i in devparents})
  for r in rows:
   q=qs[r['unit_id']];assert r['split']==split and r['arm'] in ['G0','G1','G2'] and digest(q['graph'])==r['query_graph_sha256']
   assert r['input_tokens']<=768 and len(r['new_token_ids'])<=32;max_gen_tokens=max(max_gen_tokens,len(r['new_token_ids']))
   assert not any(k in q for k in ['caption','phrase','reference','human_references'])
   assert r['parent_image_id']==q['parent_image_id'] and r['parent_image_id'] not in checkparents
   if r['arm']=='G0' and split=='train':
    old=baseline[r['unit_id']]
    for k in ['caption','new_token_ids','input_tokens','prompt_sha256','query_graph_sha256','empty_evidence','hit_token_cap']:assert r[k]==old[k],k
    assert r['parent_reused']
   elif r['arm']=='G1':assert literal_first(q['graph'])['caption']==r['caption'] and not r['new_token_ids']
   elif r['arm']=='G2':
    realized=parse_and_render(r['raw_selection_output'],q['graph']);assert realized['caption']==r['caption'] and realized['choice']==r['selection'] and realized['status']==r['status']
  assert gc['new_model_calls']<=(800 if split=='train' else 512);new_calls+=gc['new_model_calls']
  scores[split]={};diagnostics[split]={}
  for arm in ['G0','G1','G2']:
   metric=read(mpath/(arm+'.json'));assert sha(mpath/(arm+'.json'))==mc['metric_sha256'][arm]
   assert metric['expected_units']==metric['evaluated_units']==expected and set(metric['unit_ids'])==set(qs) and len(metric['per_unit'])==expected
   assert metric['metric_sources']==old_metric['metric_sources']
   local=[r for r in rows if r['arm']==arm];assert {r['unit_id'] for r in local}==set(qs)
   assert metric['empty_predictions']==sum(not r['caption'].strip() for r in local)
   assert metric['G2_invalid_selection_abstentions']==sum(r['status']=='invalid_selection_abstention' for r in local)
   assert metric['SPICE_status']=='NOT_RUN_GROUNDING_DIAGNOSTIC'
   assert metric['aggregate_native']==mc['native_scores'][arm] and metric['aggregate_display_x100']==mc['display_scores_x100'][arm]
   if split=='train' and arm=='G0':
    assert sha(mpath/'G0_original_parent_L2_metric.json')==config['parent_files_sha256']['train_metrics/L2.json']
    assert metric['aggregate_native']==old_metric['aggregate_native'] and metric['per_unit']==old_metric['per_unit'] and metric['G0_TRAIN_original_metric_reused_not_rescored']
   scores[split][arm]=metric['aggregate_display_x100'];diagnostics[split][arm]=mc['diagnostics'][arm]
   corpus_checks.append({'split':split,'arm':arm,'expected_and_evaluated':expected,'empty_predictions':metric['empty_predictions'],'native_metric_sources_match':True,'metric_reused_not_rescored':split=='train' and arm=='G0'})
  gm={r['unit_id']:r for r in rows if r['arm']=='G1'};g2=[r for r in rows if r['arm']=='G2']
  selection_stats[split]={'G2_outputs_identical_to_G1':sum(r['caption']==gm[r['unit_id']]['caption'] for r in g2),'G2_selected_existing_edge':sum(bool(r['selection']) and 'edge_id' in r['selection'] for r in g2),'G2_selected_existing_node':sum(bool(r['selection']) and 'node_id' in r['selection'] for r in g2),'G2_selected_first_edge':sum(r['selection']=={'edge_id':'e0'} for r in g2),'G2_invalid_selection_abstentions':gc['G2_invalid_selection_abstentions'],'units':expected}
  all_rows[split]=rows
  assert sha(mpath/'complete.json')==complete['TRAIN_metric_complete_sha256' if split=='train' else 'dev_metric_complete_sha256']
 assert new_calls==complete['new_model_calls']==1312 and complete['G0_TRAIN_reused']==800
 selection=read(out/'selection_from_dev.json');assert sha(out/'selection_from_dev.json')==complete['selection_sha256']
 selected=max(['G0','G1','G2'],key=lambda a:scores['dev'][a]['METEOR']);assert selected==selection['selected_arm']==complete['selected_arm_from_dev']=='G0'
 assert selection['dev_metrics_completion_sha256']==sha(out/'dev_metrics/complete.json') and selection['source_lock_sha256']==launch['preregistration_sha256']
 assert selection['automatic_check_or_TEST_launch'] is False and selection['policy_adopted'] is False
 sample=read(out/'preflight/fixed20_dev_ids.json');packet=read(out/'dev_metrics/fixed20_dev_support_packet.json');assert len(sample)==len(packet)==20 and {r['unit_id'] for r in packet}==set(sample)
 assert all(r['human_labels_created']==0 for r in packet)
 stages=read(out/'stages_complete.json');assert stages['outcomes']==complete['outcomes'] and len(stages['outcomes'])==4
 for outcome in stages['outcomes']:
  assert outcome['status']=='COMPLETE' and outcome['returncode']==0 and outcome['guard_reason'] is None
  assert outcome['peak_owned_tree_RSS_bytes']<=outcome['own_tree_RSS_cap_GiB']*1024**3 and outcome['peak_owned_GPU_bytes']<=outcome['own_GPU_cap_GiB']*1024**3 and outcome['seconds']<=outcome['wall_seconds']
 assert complete['check_or_TEST_opened'] is False and complete['automatic_check_or_TEST_launch'] is False and complete['policy_adopted'] is False
 result={'status':'PASS_R1_SOURCE_PREREG_INPUT_G0_REUSE_AND3168_TRAIN_DEV_RECORDS_SIX_NATIVE_CORPORA','utc':datetime.now(timezone.utc).isoformat(),'phase':'vg_grounding_train_dev_r1_20261103','archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,'source_manifest_sha256':sha(out/'source_manifest.json'),'preregistration_sha256':sha(out/'preregistration.json'),'sources_verified':len(manifest['source_sha256']),'parent_input_files_verified':len(config['parent_files_sha256']),'dev_input_files_verified':len(config['dev_cache_files_sha256']),'G0_TRAIN_exact_reused_captions':800,'G0_TRAIN_original_metric_reused':True,'TRAIN_records':2400,'dev_records':768,'TRAIN_units_per_arm':800,'dev_images':64,'dev_units_per_arm':256,'new_model_calls':new_calls,'native_corpora':corpus_checks,'selected_arm_from_dev':selected,'selection_stats':selection_stats,'scores_display_x100':scores,'diagnostics':diagnostics,'max_generated_token_ids':max_gen_tokens,'all_four_stage_outcomes_within_caps':True,'policy_adopted':False,'check_or_TEST_opened':False,'human_factuality_accuracy_established':False,'SPICE':'NOT_RUN','source_images_viewed':0,'metric_recomputed_by_verifier':False,'local_output_directory':out.relative_to(ROOT).as_posix()}
 savepath=ROOT/'reports/VG_GROUNDING_TRAIN_DEV_R1_COMPLETION_VERIFICATION_20261103.json';savepath.write_text(json.dumps(result,indent=2),encoding='utf8')
 with (out/'comparison.csv').open('w',newline='',encoding='utf8') as f:
  writer=csv.writer(f);writer.writerow(['split','arm','units','METEOR_x100','CIDEr_x100','ROUGE_L_x100','empty_predictions','G2_invalid_abstentions'])
  for split,expected in [('train',800),('dev',256)]:
   for arm in ['G0','G1','G2']:writer.writerow([split,arm,expected,*[scores[split][arm][k] for k in ['METEOR','CIDEr','ROUGE_L']],diagnostics[split][arm]['empty_predictions'],diagnostics[split][arm]['G2_invalid_selection_abstentions']])
 print(json.dumps({k:result[k] for k in ['status','scores_display_x100','selection_stats','selected_arm_from_dev','new_model_calls']}))
if __name__=='__main__':main()
