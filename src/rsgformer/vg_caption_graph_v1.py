"""Predicted-graph-only region views and TRAIN-only static/dynamic demonstrations."""
import hashlib,json,math

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def hash_order(namespace,identity):
    return hashlib.sha256((namespace+'|'+str(identity)).encode()).hexdigest()

def compact_graph(prediction,objects,predicates):
    """Keep the saved record external; derive explicit NMS-equivalence representatives."""
    boxes=prediction['pred_boxes'];classes=prediction['pred_classes'];scores=prediction['obj_scores']
    groups=prediction['query_to_nms_equivalence_id'];representatives={}
    for i,g in enumerate(groups):
        if g not in representatives or scores[i]>scores[representatives[g]]:representatives[g]=i
    nodes=[]
    for g,i in sorted(representatives.items()):
        b=boxes[i]
        if len(b)!=4 or not all(math.isfinite(float(x)) for x in b) or b[2]<=b[0] or b[3]<=b[1]:continue
        nodes.append({'id':'g'+str(g),'query_id':i,'class_id':int(classes[i])+1,
            'label':objects[int(classes[i])+1],'bbox_xyxy':b,'object_score':scores[i],
            'equivalent_query_ids':[j for j,v in enumerate(groups) if v==g]})
    valid={n['id'] for n in nodes};edges=[];seen=set()
    for rank,((s,o),rs,score) in enumerate(zip(prediction['pred_rel_inds'],prediction['rel_scores'],prediction['ranking_scores'])):
        s,o='g'+str(groups[s]),'g'+str(groups[o]);p=max(range(len(rs)),key=rs.__getitem__)+1
        key=(s,p,o)
        if s==o or s not in valid or o not in valid or score<=0 or not math.isfinite(score) or key in seen:continue
        seen.add(key);edges.append({'subject':s,'predicate':predicates[p],'object':o,'rank':rank,'score':score})
    return {'nodes':nodes,'edges':edges,'source':'frozen_Context1729_native_predictions',
        'representative_rule':'maximum original object score per native NMS equivalence group; original source/query IDs retained',
        'predicted_not_gt_complete':True}

def overlap_fraction(box,region):
    area=(box[2]-box[0])*(box[3]-box[1])
    return max(0,min(box[2],region[2])-max(box[0],region[0]))*max(0,min(box[3],region[3])-max(box[1],region[1]))/area if area>0 else 0

def focus_graph(graph,region,max_nodes=24,max_edges=32):
    available={n['id']:n for n in graph['nodes'] if overlap_fraction(n['bbox_xyxy'],region)>=.5}
    eligible=[e for e in graph['edges'] if e['subject'] in available and e['object'] in available]
    order=[]
    for e in eligible:
        for identity in [e['subject'],e['object']]:
            if identity not in order:order.append(identity)
    for n in sorted(available.values(),key=lambda n:(-n['object_score'],n['id'])):
        if n['id'] not in order:order.append(n['id'])
    order=order[:max_nodes];ids=set(order)
    return {'nodes':[available[i] for i in order],
        'edges':[e for e in eligible if e['subject'] in ids and e['object'] in ids][:max_edges]}

def fact_text(graph,include_color=False):
    def noun(n):
        colors=n.get('accepted_colors',[]) if include_color else []
        return (' and '.join(colors)+' ' if colors else '')+n['label']
    labels={n['id']:noun(n) for n in graph['nodes']}
    if graph['edges']:
        e=graph['edges'][0]
        return labels[e['subject']]+' '+e['predicate']+' '+labels[e['object']]
    return next(iter(labels.values()),'')

def serialize_graph(graph,include_color=False):
    nodes=[]
    for n in graph['nodes']:
        row={'id':n['id'],'object':n['label']}
        if include_color and n.get('accepted_colors'):row['color']=n['accepted_colors']
        nodes.append(row)
    return json.dumps({'objects':nodes,'relations':[[e['subject'],e['predicate'],e['object']] for e in graph['edges']]},separators=(',',':'))

def features(graph):
    labels={n['id']:n['label'] for n in graph['nodes']}
    triples={(labels[e['subject']],e['predicate'],labels[e['object']]) for e in graph['edges']}
    return set(labels.values()),triples

def retrieve_examples(query,bank,parent_image_id,static_ids,dynamic=False,k=2):
    """No caption/reference/color appears in retrieval keys; ordered SPO remains ordered."""
    qobjects,qtriples=features(query)
    def rank(example):
        objects,triples=features(example['graph'])
        if qtriples&triples:tier=0
        elif any((a==d and b==e) or (b==e and c==f) for a,b,c in qtriples for d,e,f in triples):tier=1
        elif any((a,c)==(d,f) for a,b,c in qtriples for d,e,f in triples):tier=2
        elif qobjects&objects:tier=3
        else:tier=4
        return tier,hash_order('VG_CAPTION_BANK_V1',example['example_id'])
    candidates=[e for e in bank if e['parent_image_id']!=parent_image_id and e.get('split')=='train']
    if any(e.get('split')!='train' for e in bank):raise ValueError('non_train_example_bank_forbidden')
    if dynamic:
        useful=sorted([e for e in candidates if rank(e)[0]<4],key=rank)
        candidates=useful+[e for key in static_ids for e in candidates if e['example_id']==key and e not in useful]
    else:candidates=[e for key in static_ids for e in candidates if e['example_id']==key]
    selected=[];parents=set()
    for e in candidates:
        if e['parent_image_id'] in parents:continue
        selected.append(e);parents.add(e['parent_image_id'])
        if len(selected)==k:break
    if len(selected)!=k:raise ValueError('insufficient_distinct_train_examples')
    return selected
