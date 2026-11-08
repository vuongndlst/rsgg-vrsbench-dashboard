"""Native corpus metrics; optional undefined SPICE is reported, never dropped units."""
import math
from .caption_metrics import validate_caption_units

def compute_metrics(references,predictions,runtime,include_spice=False):
    ids=validate_caption_units(references,predictions);ordinals=list(range(1,len(ids)+1));modules=runtime['modules']
    raw_refs={i:[{'caption':s} for s in references[k]] for i,k in zip(ordinals,ids)}
    raw_preds={i:[{'caption':predictions[k]}] for i,k in zip(ordinals,ids)}
    tokenizer=modules['PTBTokenizer'].PTBTokenizer();gts,res=tokenizer.tokenize(raw_refs),tokenizer.tokenize(raw_preds)
    assert set(gts)==set(res)==set(ordinals)
    aggregate={};per_unit={k:{} for k in ids}
    def save(name,value,values):
        value=float(value);values=list(map(float,values));assert math.isfinite(value) and len(values)==len(ids) and all(map(math.isfinite,values)),name
        aggregate[name]=value
        for identity,score in zip(ids,values):per_unit[identity][name]=score
    values,rows=modules['BLEU'].Bleu(4).compute_score(gts,res)
    for j in range(4):save('BLEU_'+str(j+1),values[j],rows[j])
    for name,scorer in [('CIDEr',modules['CIDEr'].Cider()),('ROUGE_L',modules['ROUGE_L'].Rouge())]:
        value,values=scorer.compute_score(gts,res);save(name,value,values)
    meteor=modules['METEOR'].Meteor()
    try:
        value,values=meteor.compute_score(gts,res);save('METEOR',value,values)
    finally:
        if meteor.lock.locked():meteor.lock.release()
        del meteor
    spice_status='NOT_REQUESTED_DEV_SELECTION'
    if include_spice:
        values=[];scorer=modules['SPICE'].Spice()
        for start in range(0,len(ids),64):
            batch=ordinals[start:start+64]
            score,rows=scorer.compute_score({i:gts[i] for i in batch},{i:res[i] for i in batch})
            assert len(rows)==len(batch)
            current=[float(row['All']['f']) for row in rows]
            if not math.isfinite(float(score)) or not all(map(math.isfinite,current)):
                spice_status='BLOCKED_UNDEFINED_NATIVE_SPICE_UNITS_NO_DROPPING';values=[];break
            values+=current
        if values:
            save('SPICE',math.fsum(values)/len(values),values);spice_status='COMPLETE_ALL_UNITS'
    return {'aggregate_native':aggregate,'aggregate_display_x100':{k:100*v for k,v in aggregate.items()},
        'per_unit':per_unit,'unit_ids':list(ids),'expected_units':len(ids),'evaluated_units':len(ids),
        'SPICE_status':spice_status,'empty_predictions':sum(not predictions[k].strip() for k in ids),
        'protocol':'native PTB complete corpus; native METEOR/BLEU/CIDEr/ROUGE; CIDEr IDF full reference corpus, never per batch',
        'metric_sources':runtime['source_hashes']}
