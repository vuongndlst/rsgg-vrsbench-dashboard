"""Grounding controls: literal graph rendering and closed existing-ID selection.
No reference text, image access, model loading, workload selection or scoring.
"""
import json,math
ARMS=('G0','G1','G2')
def validate_graph(graph):
    assert set(graph)=={'nodes','edges'},'graph_schema'
    nodes=graph['nodes'];edges=graph['edges'];assert len(nodes)<=24 and len(edges)<=32
    ids=[n['id'] for n in nodes];assert len(ids)==len(set(ids))
    assert all(isinstance(i,str) and i for i in ids)
    for n in nodes:
        assert isinstance(n['label'],str) and n['label'].strip()
        assert not n.get('candidate_colors') and not n.get('accepted_colors')
    for e in edges:
        assert e['subject'] in ids and e['object'] in ids
        assert isinstance(e['predicate'],str) and e['predicate'].strip()
    return {n['id']:n['label'] for n in nodes}

def render_choice(graph,kind,identity):
    labels=validate_graph(graph)
    if kind=='node_id':
        assert identity in labels,'unknown_node_id'
        return labels[identity]
    assert kind=='edge_id' and isinstance(identity,str)
    assert identity.startswith('e') and identity[1:].isdigit()
    i=int(identity[1:]);assert identity=='e'+str(i) and i<len(graph['edges']),'unknown_edge_id'
    e=graph['edges'][i]
    return labels[e['subject']]+' '+e['predicate']+' '+labels[e['object']]

def literal_first(graph):
    validate_graph(graph)
    if graph['edges']:return {'choice':{'edge_id':'e0'},'caption':render_choice(graph,'edge_id','e0'),'status':'rendered_existing_edge'}
    if graph['nodes']:
        identity=graph['nodes'][0]['id'];return {'choice':{'node_id':identity},'caption':render_choice(graph,'node_id',identity),'status':'rendered_existing_node'}
    return {'choice':None,'caption':'','status':'empty_evidence_abstention'}

def parse_and_render(text,graph):
    validate_graph(graph)
    if not graph['nodes']:return {'choice':None,'caption':'','status':'empty_evidence_abstention'}
    def pairs(items):
        assert len({k for k,v in items})==len(items),'duplicate_json_key'
        return dict(items)
    try:
        value=json.loads(text,object_pairs_hook=pairs,parse_constant=lambda x: (_ for _ in ()).throw(ValueError('nonfinite_json')))
        assert type(value) is dict and len(value)==1,'one_selection_key_required'
        kind,identity=next(iter(value.items()));assert kind in {'node_id','edge_id'} and type(identity) is str
        caption=render_choice(graph,kind,identity)
        return {'choice':value,'caption':caption,'status':'rendered_existing_choice'}
    except (ValueError,TypeError,AssertionError,IndexError,KeyError):
        return {'choice':None,'caption':'','status':'invalid_selection_abstention'}

def choice_messages(graph):
    validate_graph(graph)
    facts={'objects':[{'node_id':n['id'],'object':n['label']} for n in graph['nodes']],
           'relations':[{'edge_id':'e'+str(i),'subject':e['subject'],'predicate':e['predicate'],'object':e['object']} for i,e in enumerate(graph['edges'])]}
    return [{'role':'system','content':'Select one useful predicted fact for this requested region using only QUERY. Return exactly one JSON object: {"edge_id":"e0"} for an existing directed relation, or {"node_id":"g0"} for an existing object. Use an ID actually present in QUERY. Do not output a caption, extra keys, explanations, markdown, new relations or attributes.'},
            {'role':'user','content':'Choose the main supported object or one useful relation. QUERY: '+json.dumps(facts,separators=(',',':'))+'\nSelection:'}]

def geometry_only_units(image_id,image_size,regions):
    assert type(image_id) is int and image_id>0
    width,height=image_size;assert all(type(x) is int and x>0 for x in image_size)
    units={};rejected=[];seen=set()
    for r in regions:
        identity=r['region_id'];assert type(identity) is int and identity>0 and identity not in seen
        assert r['image_id']==image_id;seen.add(identity)
        uid='vg:'+str(image_id)+':region:'+str(identity)
        values=[r[k] for k in ('x','y','width','height')];reason=None
        if any(type(x) not in (int,float) or not math.isfinite(x) for x in values):reason='invalid_geometry'
        else:
            x,y,w,h=values
            if x<0 or y<0 or w<=0 or h<=0 or not math.isfinite(x+w) or not math.isfinite(y+h) or x+w>width or y+h>height:reason='geometry_not_fully_contained'
        if reason:rejected.append({'unit_id':uid,'reason':reason});continue
        x,y,w,h=values
        units[uid]={'unit_id':uid,'parent_image_id':image_id,'region_id':identity,'image_size':[width,height],'crop_xyxy':[math.floor(x),math.floor(y),math.ceil(x+w),math.ceil(y+h)],'scope':'provided_crop_conditional_caption','split':'val','partition':'grounding_fresh_dev_v1'}
    assert len(units)+len(rejected)==len(regions)
    return units,rejected

def synthetic_contract():
    g={'nodes':[{'id':'g0','label':'leaf'},{'id':'g1','label':'flower'}],'edges':[{'subject':'g0','predicate':'on','object':'g1'}]}
    assert literal_first(g)['caption']=='leaf on flower'
    assert parse_and_render('{"edge_id":"e0"}',g)['caption']=='leaf on flower'
    assert parse_and_render('{"node_id":"g1"}',g)['caption']=='flower'
    cases=['{"edge_id":"e1"}','{"edge_id":"e00"}','{"node_id":"missing"}','{"node_id":0}','{"node_id":"g0","edge_id":"e0"}','{"node_id":"g0","node_id":"g1"}','{"caption":"wooden flower"}','```json\n{"node_id":"g0"}\n```','{"node_id":"g0"} extra','[]','null','NaN']
    for t in cases:assert parse_and_render(t,g)['status']=='invalid_selection_abstention'
    reversed_g={'nodes':g['nodes'],'edges':[{'subject':'g1','predicate':'on','object':'g0'}]}
    assert literal_first(reversed_g)['caption']=='flower on leaf'
    assert literal_first({'nodes':list(reversed(g['nodes'])),'edges':g['edges']})['caption']=='leaf on flower'
    assert literal_first({'nodes':[g['nodes'][1]],'edges':[]})['caption']=='flower'
    assert literal_first({'nodes':[],'edges':[]})['caption']==''
    geometry={'region_id':10,'image_id':1,'x':1.2,'y':2,'width':3,'height':4,'phrase':'do not use this reference'}
    u,_=geometry_only_units(1,[10,10],[geometry]);geometry['phrase']='different reference'
    assert geometry_only_units(1,[10,10],[geometry])[0]==u
    del geometry['phrase'];assert geometry_only_units(1,[10,10],[geometry])[0]==u
    assert not geometry_only_units(1,[10,10],[{**geometry,'x':20}])[0]
    return {'status':'SYNTHETIC_GROUNDING_CONTRACT_PASS','invalid_selection_cases':len(cases),'no_inverse_edge_or_reference_correction':True,'empty_invalid_outputs_retained':True,'reference_independent_geometry':True}
