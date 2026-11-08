"""Complete-corpus scoring of100provided VAL regions; no global GT invented."""
from pathlib import Path
import collections,gzip,json,os,re,signal,traceback
ROOT=Path.home()/'rsg-former';PIPE=ROOT/'artifacts/vg_caption_pilot100_v1_20261026';ART=PIPE/'metrics';GEN=PIPE/'generation';BASE=ROOT/'artifacts/vg_caption_execution_v1_20261025'
os.sched_setaffinity(0,set(sorted(os.sched_getaffinity(0))[:4]));signal.alarm(2700)
os.environ.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
from rsgformer.vg_color_execution_common_v1 import sha,save,line,now

def main():
 assert not ART.exists();ART.mkdir()
 lock=json.loads((PIPE/'preregistration.json').read_text())
 for p,h in lock['source_sha256'].items():assert sha(ROOT/p)==h
 complete=json.loads((GEN/'complete.json').read_text());assert sha(GEN/'generation_commit.jsonl')==complete['generation_sha256']
 assert sha(GEN/'query_units_bounded.json')==complete['query_units_sha256']
 units=json.loads((GEN/'query_units_bounded.json').read_text());ids={u['unit_id'] for u in units};assert len(ids)==100
 source=json.loads((BASE/'base_cache_complete.json').read_text());assert sha(BASE/'human_references_separate.json')==source['file_sha256']['human_references_separate.json']
 all_refs=json.loads((BASE/'human_references_separate.json').read_text());refs={i:all_refs[i] for i in ids};assert all(refs.values())
 save(ART/'human_references_original.json',refs)
 save(ART/'preregistration.json',{'utc':now(),'parent_lock_sha256':sha(PIPE/'preregistration.json'),'generation_sha256':complete['generation_sha256'],
  'human_reference_source_sha256':sha(BASE/'human_references_separate.json'),'expected_units_per_corpus':100,'same_references_all12corpora':True,
  'native_corpus_CIDEr_global_IDF':True,'SPICE':'not_run_initial_pilot','automatic_prompt_or_TEST_selection':False})
 java=json.loads((ROOT/'reports/CAPTION_JAVA_PREPARATION.json').read_text());j=Path(java['java_path_alias'].replace('${REMOTE_HOME}',str(Path.home())))
 os.environ['PATH']=str(j.parent)+os.pathsep+os.environ['PATH'];os.environ['_JAVA_OPTIONS']='-Xmx2g -XX:ActiveProcessorCount=1 -XX:+UseSerialGC -XX:MaxMetaspaceSize=256m -XX:ReservedCodeCacheSize=128m'
 from rsgformer.caption_metric_runtime import private_metric_runtime
 from rsgformer.caption_metrics_v2 import compute_metrics
 from rsgformer.vg_caption_candidate_prompt_v2 import ARMS
 rows=list(map(json.loads,(GEN/'generation_commit.jsonl').read_text().splitlines()));assert len(rows)==1200
 results={};queries={u['unit_id']:u for u in units};palette=set('black white gray grey red green blue yellow brown orange pink purple'.split())
 for template in ['P1','P2']:
  for arm in ARMS:
   local=[r for r in rows if r['template']==template and r['arm']==arm];assert len(local)==100 and {r['unit_id'] for r in local}==ids
   scratch=ROOT/'artifacts/caption_readiness'/('pilot100_20261026_'+template+'_'+arm);scratch.mkdir(exist_ok=False)
   with private_metric_runtime(ROOT,scratch) as runtime:metrics=compute_metrics(refs,{r['unit_id']:r['caption'] for r in local},runtime,include_spice=False)
   flags=0
   for r in local:
    words=set(re.findall('[a-z]+',r['caption'].lower()));supplied=set()
    for n in queries[r['unit_id']]['graph']['nodes']:
     if arm.startswith('C'):supplied.update(n.get('candidate_colors',[]))
     if n['label'] in palette:supplied.add(n['label'])
    mentioned={('gray' if w=='grey' else w) for w in words&palette}
    flags+=bool(mentioned-supplied)
   metrics.update(template=template,arm=arm,images=100,SPICE_status='NOT_RUN_INITIAL_PILOT',
    hit_output_cap=sum(r['hit_token_cap'] for r in local),no_evidence_units=sum(r['no_evidence'] for r in local),
    lexical_color_mentions_without_query_support_flags=flags,lexical_flags_are_not_human_factuality_accuracy=True,
    color_scope='unverified_candidate_only' if arm.startswith('C') else 'no_color',provided_region=True,TEST_scored=False)
   save(ART/(template+'_'+arm+'.json'),metrics);results[template+'_'+arm]=metrics
   save(ART/'progress.json',{'utc':now(),'corpora':len(results),'expected':12,'last':template+'_'+arm})
 samples=[]
 for u in sorted(units,key=lambda u:u['unit_id'])[:20]:
  samples.append({'unit_id':u['unit_id'],'parent_image_id':u['parent_image_id'],'crop_xyxy':u['crop_xyxy'],'references':refs[u['unit_id']],
   'graph':u['graph'],'outputs':{r['template']+'_'+r['arm']:r['caption'] for r in rows if r['unit_id']==u['unit_id']}})
 save(ART/'qualitative_examples_fixed_first20.json',samples)
 summary={'status':'PILOT100_ALL12_CONFIG_CORPUS_METRICS_COMPLETE','utc':now(),'images':100,'provided_regions':100,'outputs':1200,
  'scores_display_x100':{k:v['aggregate_display_x100'] for k,v in results.items()},
  'diagnostics':{k:{f:v[f] for f in ['empty_predictions','no_evidence_units','hit_output_cap','lexical_color_mentions_without_query_support_flags']} for k,v in results.items()},
  'SPICE_status':'NOT_RUN_INITIAL_PILOT','automatic_selection_or_TEST_launch':False,'human_color_precision_gate_passed':False,
  'limitations':['VAL pilot100images,one provided region per image,one original human phrase','predicted SG and candidate colors may be incorrect or missing',
   'curated TRAIN fact-template demonstrations are not new human labels','not official5k VG benchmark or automatic dense captioning','no image-global human GT fabricated','not caption-clean weights or SOTA claim'],
  'metrics_file_sha256':{k+'.json':sha(ART/(k+'.json')) for k in results}}
 save(ART/'complete.json',summary);save(ROOT/'reports/VG_CAPTION_PILOT100_RESULT_20261026.json',summary)
 lines=['# Caption pilot100VAL images: one provided region per image','','All metrics below use native corpus evaluation and display scaling x100. This is a pilot, not matched SOTA.','','| Configuration | METEOR | CIDEr | BLEU4 | ROUGE-L |','|---|---:|---:|---:|---:|']
 for k,v in summary['scores_display_x100'].items():lines.append('| '+k+' | '+' | '.join(format(v[n],'.3f') for n in ['METEOR','CIDEr','BLEU_4','ROUGE_L'])+' |')
 lines+=['','N: no color; C: unverified candidate colors. Z: zero-shot; S: staticK2; D: dynamicK2. P1: noun phrase; P2: concise sentence.','No automatic winner selection or TEST launch. SPICE not run. Empty outputs remain in every corpus.']
 (ART/'comparison.md').write_text('\n'.join(lines));print(line(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as e:
  if ART.exists():save(ART/'failure.json',{'utc':now(),'reason':str(e),'TEST_accessed':False})
  traceback.print_exc();raise
