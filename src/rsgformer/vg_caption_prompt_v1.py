"""Deterministic same-query serialization, paired demonstrations and arm prompts."""
import copy
from .vg_caption_graph_v1 import serialize_graph,fact_text,retrieve_examples,hash_order

ARMS=['N-Z','N-S','N-D','C-Z','C-S','C-D']
SYSTEM=('Verbalize only the predicted visual facts in QUERY. EXAMPLES are independent scenes; never transfer their colors, objects or relations to QUERY. '
 'Use color only if explicitly supplied for that object. Do not invent missing actions, materials, counts, identities or intentions. '
 'Omit unknown details; do not add scene, place or time absent from QUERY. Describe only the requested region. Return only the description, without IDs, scores or explanation.')

def bounded_query_graph(graph,tokenizer):
    graph=copy.deepcopy(graph)
    while len(tokenizer.encode(serialize_graph(graph,False),add_special_tokens=False))>320:
        if graph['edges']:graph['edges'].pop()
        else:graph['nodes'].pop()
    # Identical base facts across all six arms; only attributes can be dropped.
    while len(tokenizer.encode(serialize_graph(graph,True),add_special_tokens=False))>448:
        colored=[n for n in graph['nodes'] if n.get('accepted_colors')]
        if not colored:raise ValueError('query_graph_budget_contract')
        colored[-1].pop('accepted_colors')
    return graph

def demo_graph(graph):
    if graph['edges']:
        edge=graph['edges'][0];ids={edge['subject'],edge['object']}
        return {'nodes':[n for n in graph['nodes'] if n['id'] in ids],'edges':[edge]}
    return {'nodes':graph['nodes'][:1],'edges':[]}

def build_bank(train_units,tokenizer,protected_train_ids=()):
    blocked=set(protected_train_ids);rows=[]
    for unit in train_units:
        assert unit['split']=='train' and unit['partition']=='train_bank'
        if unit['parent_image_id'] in blocked or not unit['graph']['nodes']:continue
        graph=demo_graph(unit['graph'])
        if any(n['label'] in {'orange'} for n in graph['nodes']):continue
        a,b=fact_text(graph,False),fact_text(graph,True)
        # Both views are curated predicted-fact templates, not new human GT.
        color_demo='EXAMPLE FACTS: '+serialize_graph(graph,True)+'\nDESCRIPTION: '+b
        plain_demo='EXAMPLE FACTS: '+serialize_graph(graph,False)+'\nDESCRIPTION: '+a
        if max(len(tokenizer.encode(s,add_special_tokens=False)) for s in [color_demo,plain_demo])>128:continue
        rows.append({'example_id':unit['unit_id'],'parent_image_id':unit['parent_image_id'],'split':'train','graph':graph,
            'caption_no_color':a,'caption_color':b,'demonstration_source':'curated deterministic supported predicted facts; not original human annotation',
            'human_reference_sha256':unit.get('human_reference_sha256')})
    rows=sorted(rows,key=lambda e:hash_order('VG_CAPTION_BANK_V1',e['example_id']))[:2000]
    if len(rows)<2:raise ValueError('insufficient_TRAIN_bank')
    static=[];parents=set()
    for require_relation in [False,True]:
        for e in rows:
            if bool(e['graph']['edges'])==require_relation and e['parent_image_id'] not in parents:
                static.append(e['example_id']);parents.add(e['parent_image_id']);break
    for e in rows:
        if e['parent_image_id'] not in parents:static.append(e['example_id']);parents.add(e['parent_image_id'])
        if len(static)>=10:break
    assert len(static)>=2
    return rows,static

def make_messages(unit,arm,template,bank,static_ids):
    color=arm.startswith('C-');policy=arm[-1];graph=unit['graph'];examples=[]
    if policy!='Z':examples=retrieve_examples(graph,bank,unit['parent_image_id'],static_ids,dynamic=policy=='D')
    demonstrations=[]
    for e in examples:
        demonstrations.append('EXAMPLE FACTS: '+serialize_graph(e['graph'],color)+'\nDESCRIPTION: '+e['caption_color' if color else 'caption_no_color'])
    instruction=('Write one short English noun phrase, usually 4-10 words; one main object and at most one useful relation.' if template=='P1' else
        'Write one concise English sentence, usually 6-12 words; describe the main supported object or relation.')
    user='\n\n'.join(demonstrations+[instruction+' Be shorter if evidence is limited.\nQUERY FACTS: '+serialize_graph(graph,color)+'\nDescription:'])
    return [{'role':'system','content':SYSTEM},{'role':'user','content':user}],[e['example_id'] for e in examples]
