"""Predicted-SG candidate-color pilot prompts; no evaluation references consumed."""
import copy,json
from .vg_caption_graph_v1 import hash_order,retrieve_examples
ARMS=['N-Z','N-S','N-D','C-Z','C-S','C-D']
SYSTEM=('Describe only predicted visual facts supplied in QUERY for the requested region. EXAMPLES describe independent scenes: never copy their colors, objects or relations into QUERY. '
 'Color candidates are uncertain attributes; use only the colors explicitly attached to that query object. Omit unknown details. '
 'Never invent actions, materials, counts, identities, locations, time or intentions. Output only the short English description, without IDs, scores or explanation.')

def serialize(graph,color):
 nodes=[]
 for n in graph['nodes']:
  row={'id':n['id'],'object':n['label']}
  if color and n.get('candidate_colors'):row['candidate_color']=n['candidate_colors']
  nodes.append(row)
 return json.dumps({'objects':nodes,'relations':[[e['subject'],e['predicate'],e['object']] for e in graph['edges']]},separators=(',',':'))

def bounded(graph,tokenizer):
 g=copy.deepcopy(graph)
 while len(tokenizer.encode(serialize(g,False),add_special_tokens=False))>256:
  if g['edges']:g['edges'].pop()
  else:g['nodes'].pop()
 while len(tokenizer.encode(serialize(g,True),add_special_tokens=False))>320:
  colored=[n for n in g['nodes'] if n.get('candidate_colors')];assert colored,'color_budget'
  colored[-1].pop('candidate_colors')
 return g

def demo_graph(graph):
 if graph['edges']:
  e=graph['edges'][0];ids={e['subject'],e['object']}
  return {'nodes':[n for n in graph['nodes'] if n['id'] in ids],'edges':[e]}
 return {'nodes':graph['nodes'][:1],'edges':[]}

def fact_text(graph,color):
 labels={n['id']:(' and '.join(n.get('candidate_colors',[]))+' ' if color and n.get('candidate_colors') else '')+n['label'] for n in graph['nodes']}
 if graph['edges']:
  e=graph['edges'][0];return labels[e['subject']]+' '+e['predicate']+' '+labels[e['object']]
 return next(iter(labels.values()),'')

def build_bank(units,tokenizer,blocked=()):
 rows=[];blocked=set(blocked)
 for u in units:
  assert u['split']=='train' and u['partition']=='train_bank'
  if u['parent_image_id'] in blocked or not u['graph']['nodes']:continue
  g=demo_graph(u['graph'])
  if any(n['label']=='orange' for n in g['nodes']):continue
  plain,colored=fact_text(g,False),fact_text(g,True)
  texts=['EXAMPLE FACTS: '+serialize(g,c)+'\nDESCRIPTION: '+(colored if c else plain) for c in [False,True]]
  if max(len(tokenizer.encode(s,add_special_tokens=False)) for s in texts)>128:continue
  rows.append({'example_id':u['unit_id'],'parent_image_id':u['parent_image_id'],'split':'train','graph':g,
   'caption_no_color':plain,'caption_color':colored,'demonstration_source':'curated deterministic predicted-fact templates, not human labels'})
 rows=sorted(rows,key=lambda e:hash_order('VG_CAPTION_BANK_V1',e['example_id']))[:2000];assert len(rows)>=2
 static=[];parents=set()
 for relation in [False,True]:
  for e in rows:
   if bool(e['graph']['edges'])==relation and e['parent_image_id'] not in parents:
    static.append(e['example_id']);parents.add(e['parent_image_id']);break
 for e in rows:
  if len(static)>=10:break
  if e['parent_image_id'] not in parents:static.append(e['example_id']);parents.add(e['parent_image_id'])
 assert len(static)>=2;return rows,static

def messages(unit,arm,template,bank,static):
 assert arm in ARMS and template in ['P1','P2']
 color=arm.startswith('C');policy=arm[-1]
 examples=retrieve_examples(unit['graph'],bank,unit['parent_image_id'],static,dynamic=policy=='D') if policy!='Z' else []
 demos=['EXAMPLE FACTS: '+serialize(e['graph'],color)+'\nDESCRIPTION: '+e['caption_color' if color else 'caption_no_color'] for e in examples]
 instruction=('Write one short noun phrase, usually 4-10 words, with one main object and at most one useful relation.' if template=='P1' else
  'Write one concise sentence, usually 6-12 words, describing the main supported object or relation.')
 content='\n\n'.join(demos+[instruction+' Be shorter if evidence is limited.\nQUERY FACTS: '+serialize(unit['graph'],color)+'\nDescription:'])
 return [{'role':'system','content':SYSTEM},{'role':'user','content':content}],[e['example_id'] for e in examples]
