"""Three preregistered geometry-only policies; no references, colors or new objects."""
from .vg_caption_graph_v1 import focus_graph
POLICIES=['L0','L1','L2']
def area(b):return max(0,b[2]-b[0])*max(0,b[3]-b[1])
def inter(a,b):return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
def select(graph,region,policy):
 assert policy in POLICIES
 original=focus_graph(graph,region,max_nodes=24,max_edges=32)
 if policy=='L0' or (policy=='L1' and original['nodes']):return original
 overlap=[n for n in graph['nodes'] if inter(n['bbox_xyxy'],region)>0]
 if not overlap:return original
 def rank(n):
  x=inter(n['bbox_xyxy'],region);iou=x/max(1e-12,area(n['bbox_xyxy'])+area(region)-x)
  return -iou,-n['object_score'],n['id']
 anchor=min(overlap,key=rank)
 if policy=='L1':return {'nodes':[anchor],'edges':[]}
 nodes=([anchor]+[n for n in original['nodes'] if n['id']!=anchor['id']])[:24];ids={n['id'] for n in nodes}
 return {'nodes':nodes,'edges':[e for e in graph['edges'] if e['subject'] in ids and e['object'] in ids][:32]}

def synthetic_contract():
 def node(identity,box,score=.5):return {'id':identity,'bbox_xyxy':box,'object_score':score,'label':'chair'}
 g={'nodes':[node('b',[0,0,100,100]),node('a',[0,0,100,100])],'edges':[]}
 assert not select(g,[40,40,60,60],'L0')['nodes']
 for p in ['L1','L2']:assert select(g,[40,40,60,60],p)['nodes'][0]['id']=='a'
 for p in POLICIES:assert not select(g,[200,200,210,210],p)['nodes']
 g={'nodes':[node('large',[0,0,100,100],.9),node('small',[40,40,60,60],.1)],'edges':[{'subject':'large','object':'small','predicate':'near'}]}
 a=select(g,[35,35,65,65],'L0');assert [n['id'] for n in a['nodes']]==['small']
 assert select(g,[35,35,65,65],'L1')==a and select(g,[35,35,65,65],'L2')['nodes'][0]['id']=='small'
 return {'status':'SYNTHETIC_GEOMETRY_CONTRACT_PASS','tiny_crop_fallback':True,'disjoint_remains_empty':True,'tie_order_deterministic':True,'legacy_nonempty_preserved_in_L1':True}
