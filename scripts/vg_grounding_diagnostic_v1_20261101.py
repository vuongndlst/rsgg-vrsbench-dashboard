"""Frozen, local-only TRAIN lexical/structural diagnosis. No inference or policy selection."""
from pathlib import Path
import collections,ctypes,gzip,hashlib,json,re,time
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
CONFIG='configs/vg_grounding_diagnostic_v1_20261101.json'
PREREG='reports/VG_GROUNDING_DIAGNOSTIC_PREREGISTRATION_20261101.json'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def tokens(s):return re.findall(r"[a-z]+|[0-9]+",s.lower())
def variants(c):
    result={c}
    if c in {'men','people','sheep','hair','food','fruit','snow'}:return result
    if c=='man':return result
    if c=='woman':return result|{'women'}
    if c=='child':return result|{'children'}
    if c=='person':return result
    if c=='leaf':return result|{'leaves'}
    if c=='shelf':return result|{'shelves'}
    if c.endswith('y') and c[-2] not in 'aeiou':return result|{c[:-1]+'ies'}
    return result|{c+('es' if c.endswith(('s','x','ch','sh')) else 's')}
def lexicon(classes):
    table=collections.defaultdict(set)
    for c in classes:
        for v in variants(c):table[tuple(tokens(v))].add(c)
    return table

def mentions(text,table):
    ts=tokens(text);found=[];i=0
    while i<len(ts):
        hit=None
        for size in range(min(4,len(ts)-i),0,-1):
            matches=table.get(tuple(ts[i:i+size]),set())
            if len(matches)==1:hit=(size,next(iter(matches)));break
        if hit:found.append({'start':i,'end':i+hit[0],'label':hit[1]});i+=hit[0]
        else:i+=1
    return ts,found

def relation_check(text,graph,table,relations):
    ts,ms=mentions(text,table)
    if not text:return {'status':'empty_caption'}
    if len(ms)!=2:return {'status':'unparsed_object_count','recognized_object_mentions':len(ms)}
    if ms[0]['label']==ms[1]['label']:return {'status':'ambiguous_same_class_endpoints'}
    middle=[t for t in ts[ms[0]['end']:ms[1]['start']] if t not in {'a','an','the'}]
    if middle and middle[0] in {'is','are'}:middle=middle[1:]
    pred=' '.join(middle)
    if pred not in relations:return {'status':'unparsed_predicate','surface':pred}
    labels={n['id']:n['label'] for n in graph['nodes']};s,o=ms[0]['label'],ms[1]['label']
    edges={(labels[e['subject']],e['predicate'],labels[e['object']]) for e in graph['edges']}
    forward=(s,pred,o) in edges;reverse=(o,pred,s) in edges
    status='possible_forward_and_reverse' if forward and reverse else 'possible_forward_edge' if forward else 'reverse_edge_only' if reverse else 'no_explicit_matching_edge'
    return {'status':status,'subject_label':s,'predicate':pred,'object_label':o,'instance_grounding_verified':False}

def fixtures(classes,relations):
    tab=lexicon(classes)
    graph={'nodes':[{'id':'a','label':'leaf'},{'id':'b','label':'flower'}],'edges':[{'subject':'a','predicate':'on','object':'b'}]}
    assert relation_check('a leaf on a flower',graph,tab,relations)['status']=='possible_forward_edge'
    assert relation_check('flowers on a leaf',graph,tab,relations)['status']=='reverse_edge_only'
    assert relation_check('leaf near flower',graph,tab,relations)['status']=='no_explicit_matching_edge'
    assert relation_check('a leaf',graph,tab,relations)['status']=='unparsed_object_count'
    assert relation_check('leaf next to leaf',graph,tab,relations)['status']=='ambiguous_same_class_endpoints'
    assert relation_check('',graph,tab,relations)['status']=='empty_caption'
    assert [m['label'] for m in mentions('women and children near shelves',tab)[1]]==['woman','child','shelf']
    assert [m['label'] for m in mentions('men and people',tab)[1]]==['men','people']
    assert not mentions('doorway',tab)[1]
    return 9

def main():
    start=time.time();assert start<1794670321.2414355,'absolute_deadline'
    kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p
    handle=kernel.GetCurrentProcess();mask=ctypes.c_size_t();system=ctypes.c_size_t()
    kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)]
    kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
    assert kernel.GetProcessAffinityMask(handle,ctypes.byref(mask),ctypes.byref(system))
    assert kernel.SetProcessAffinityMask(handle,mask.value & -mask.value)
    cfg=json.loads((ROOT/CONFIG).read_text());pr=json.loads((ROOT/PREREG).read_text())
    for rel,h in {**pr['input_sha256'],**pr['source_config_sha256']}.items():assert sha(ROOT/rel)==h,rel
    assert cfg['selection_or_policy_adoption_allowed'] is False
    classes=cfg['ontology'];table=lexicon(classes);fixture_count=fixtures(classes,cfg['literal_predicates'])
    out=ROOT/cfg['output_directory'];out.mkdir(exist_ok=False)
    p=ROOT/cfg['parent_directory'];query=json.loads((p/'train_generation/query_units_by_policy.json').read_text());refs=json.loads((p/'train_metrics/original_human_references.json').read_text())
    with gzip.open(p/'prepare/train_graphs.jsonl.gz','rt') as f:full={v['image_id']:v['graph'] for v in map(json.loads,f)}
    with (p/'train_generation/generation_commit.jsonl').open() as f:gen=list(map(json.loads,f))
    assert len(full)==200 and len(gen)==2400 and len(refs)==800
    outputs={(g['policy'],g['unit_id']):g for g in gen};assert len(outputs)==2400
    sets=[{u['unit_id'] for u in query[k]} for k in ['L0','L1','L2']];assert sets[0]==sets[1]==sets[2]==set(refs)
    fixed20=sorted(sets[0],key=lambda u:hashlib.sha256(('VG_GROUNDING_TRAIN_FIXED20_V1|'+u).encode()).hexdigest())[:20]
    rows=[];policy_summary={};fixed=[]
    for policy in ['L0','L1','L2']:
        for u in query[policy]:
            assert u['split']=='train' and u['partition']=='train_bank'
            uid=u['unit_id'];g=u['graph'];original=full[u['parent_image_id']];record=outputs[policy,uid]
            assert digest(g)==record['query_graph_sha256']
            assert digest(original)==u['base_graph_sha256']
            fg={n['label'] for n in original['nodes']};qg={n['label'] for n in g['nodes']}
            rt=sorted({m['label'] for text in refs[uid] for m in mentions(text,table)[1]})
            caption=record['caption'];ts,cm=mentions(caption,table);ct=sorted({m['label'] for m in cm})
            sg_text=' '.join([n['label'] for n in g['nodes']]+[e['predicate'] for e in g['edges']]);sg_tokens=set(tokens(sg_text))
            used_positions={j for m in cm for j in range(m['start'],m['end'])}
            extras={kind:sorted({t for j,t in enumerate(ts) if t in vocabulary and j not in used_positions and t not in sg_tokens}) for kind,vocabulary in cfg['attribute_flag_lexicons'].items()}
            counts=sorted({t for t in ts if t in cfg['count_words'] or t.isdigit()})
            r={'policy':policy,'unit_id':uid,'parent_image_id':u['parent_image_id'],'empty_graph':not g['nodes'],'empty_caption':not caption.strip(),'node_count':len(g['nodes']),'edge_count':len(g['edges']),
               'reference_lexical_classes':rt,'reference_classes_missing_full_graph':sorted(set(rt)-fg),'reference_classes_present_full_missing_focus':sorted((set(rt)&fg)-qg),'reference_classes_present_focus':sorted(set(rt)&qg),
               'caption':caption,'caption_lexical_classes':ct,'caption_classes_absent_query':sorted(set(ct)-qg),'attribute_surface_flags_not_in_query':extras,'physical_count_not_verified':counts,
               'relation_surface_check':relation_check(caption,g,table,cfg['literal_predicates'])}
            rows.append(r)
            if uid in fixed20:fixed.append({**r,'references_evaluation_only':refs[uid],'query_graph':g})
        rr=[r for r in rows if r['policy']==policy]
        policy_summary[policy]={'all_units':len(rr),'empty_graphs':sum(r['empty_graph'] for r in rr),'empty_captions':sum(r['empty_caption'] for r in rr),'units_with_any_ontology_reference_term':sum(bool(r['reference_lexical_classes']) for r in rr),
            'units_with_all_matched_reference_terms_missing_full_graph':sum(bool(r['reference_lexical_classes']) and len(r['reference_classes_missing_full_graph'])==len(r['reference_lexical_classes']) for r in rr),
            'units_with_any_reference_term_missing_full_graph':sum(bool(r['reference_classes_missing_full_graph']) for r in rr),'units_with_any_reference_term_present_full_missing_focus':sum(bool(r['reference_classes_present_full_missing_focus']) for r in rr),
            'units_with_any_reference_term_in_focus':sum(bool(r['reference_classes_present_focus']) for r in rr),'units_with_caption_class_absent_query':sum(bool(r['caption_classes_absent_query']) for r in rr),
            'attribute_flagged_outputs':{k:sum(bool(r['attribute_surface_flags_not_in_query'][k]) for r in rr) for k in cfg['attribute_flag_lexicons']},
            'outputs_with_unverified_count_word':sum(bool(r['physical_count_not_verified']) for r in rr),'relation_surface_status':dict(collections.Counter(r['relation_surface_check']['status'] for r in rr))}
    by={(r['policy'],r['unit_id']):r for r in rows};paired={}
    for policy in ['L1','L2']:
        changed=[uid for uid in sorted(sets[0]) if by['L0',uid]['caption']!=by[policy,uid]['caption']]
        paired[policy]={'paired_units':800,'captions_changed':len(changed),'changed_from_L0_empty':sum(by['L0',uid]['empty_caption'] for uid in changed),'changed_from_L0_nonempty':sum(not by['L0',uid]['empty_caption'] for uid in changed)}
    with (out/'unit_diagnostics.jsonl').open('x',encoding='utf8') as f:
        for r in rows:f.write(json.dumps(r,sort_keys=True)+'\n')
    (out/'fixed20_TRAIN_packet.json').write_text(json.dumps({'sampling':'fixed hash order before counters,20TRAIN units x3policies','unit_ids':fixed20,'rows':fixed},indent=2),encoding='utf8')
    summary={'status':'COMPLETE_TRAIN_LEXICAL_STRUCTURAL_DIAGNOSIS_NOT_VISUAL_ACCURACY','utc':datetime.now(timezone.utc).isoformat(),'phase':cfg['phase'],'preregistration_sha256':sha(ROOT/PREREG),'input_sha256':pr['input_sha256'],'source_config_sha256':pr['source_config_sha256'],
        'TRAIN_images':len(full),'TRAIN_regions':800,'saved_caption_records':2400,'policy_summary':policy_summary,'paired_changes':paired,'synthetic_contract_cases_passed':fixture_count,
        'query_reference_used_as_generator_input':False,'new_inference_or_metric_scoring':False,'policy_selected_or_adopted':False,'TEST_images_predictions_or_references_read':False,'source_images_viewed':0,'human_labels_created':0,
        'limits':cfg['limitations'],'wall_seconds':time.time()-start,'outputs_sha256':{f.name:sha(f) for f in out.iterdir() if f.is_file()}}
    for rel,h in pr['input_sha256'].items():assert sha(ROOT/rel)==h,'input_modified'
    assert time.time()-start<75 and sum(f.stat().st_size for f in out.iterdir())<32*1024**2
    (out/'complete.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    (ROOT/'reports/VG_GROUNDING_DIAGNOSTIC_RESULT_20261101.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps({'status':summary['status'],'TRAIN_regions':800,'saved_caption_records':2400,'policy_summary':policy_summary,'paired_changes':paired,'wall_seconds':summary['wall_seconds']}))
if __name__=='__main__':main()
