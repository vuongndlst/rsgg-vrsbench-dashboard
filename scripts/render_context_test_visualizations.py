# -*- coding: utf-8 -*-
"""Render saved model detections and predicted scene graphs as local research figures."""
from pathlib import Path
import html,json,math,os,shutil,subprocess,zipfile
from PIL import Image,ImageDraw,ImageFont,ImageOps

ROOT=Path(__file__).absolute().parents[1];OUT=ROOT/'outputs/context_1729_test_visualization_20261024'
COLORS=['#007F86','#2457BE','#C25B0D','#98359B','#398347','#C13D55','#7152C3','#927023','#457C9E','#626B26']
FONTS=Path('C:/Windows/Fonts')
def font(n,bold=False):return ImageFont.truetype(str(FONTS/('segoeuib.ttf' if bold else 'segoeui.ttf')),n)
def fit(im,size):return ImageOps.contain(im,size,Image.Resampling.LANCZOS)
def paste_center(canvas,im,rect):
 x,y,w,h=rect;small=fit(im,(w,h));canvas.paste(small,(x+(w-small.width)//2,y+(h-small.height)//2));return small
def jsonwrite(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf8')

payload=json.loads((OUT/'samples.json').read_text(encoding='utf8'));objects=payload['object_names'];predicates=payload['predicate_names'];gallery=[]
assert len(payload['samples'])==6
for record in payload['samples']:
 identity=record['image_id'];stem='vg_'+str(identity);pred=record['prediction'];eq=pred['query_to_nms_equivalence_id'];nodes=[];edges=[];triples=set()
 for rank,((s,o),scores,score) in enumerate(zip(pred['pred_rel_inds'],pred['rel_scores'],pred['ranking_scores']),1):
  sid,oid=int(eq[s]),int(eq[o]);label=max(range(len(scores)),key=lambda k:scores[k])+1
  if sid==oid or score<=0 or (sid,oid,label) in triples:continue
  new=[x for x in (sid,oid) if x not in nodes]
  if len(set(nodes+new))>10:continue
  for x in new:
   if x not in nodes:nodes.append(x)
  triples.add((sid,oid,label));edges.append({'subject_group':sid,'object_group':oid,'subject_query':s,'object_query':o,'predicate_id':label,'predicate':predicates[label],'predicate_score':scores[label-1],'ranking_score':score,'original_rank':rank})
  if len(edges)==12:break
 assert nodes and edges
 display={x:i+1 for i,x in enumerate(nodes)};node_records=[]
 for group in nodes:
  q=max((q for q,e in enumerate(eq) if e==group),key=lambda q:pred['obj_scores'][q]);label=int(pred['pred_classes'][q])+1
  node_records.append({'id':display[group],'nms_group':group,'representative_query':q,'class_id':label,'label':objects[str(label)],'box_xyxy':pred['pred_boxes'][q],'raw_object_score':pred['obj_scores'][q],'color':COLORS[(display[group]-1)%len(COLORS)]})
 for e in edges:e.update(subject_id=display[e['subject_group']],object_id=display[e['object_group']])
 byid={n['id']:n for n in node_records}
 visual={'image_id':identity,'test_index':record['test_index'],'checkpoint_sha256':payload['checkpoint_sha256'],'seed':1729,'selected_epoch':10,
         'display_rule':'First12 positive-ranking non-self NMS-group triplets within10groups; no GT/correctness filtering. Display representative is highest object-score query within group. Raw predictions unchanged.',
         'nodes':node_records,'edges':edges,'predicted_queries_total':len(pred['pred_boxes']),'saved_ranked_relations_total':len(pred['pred_rel_inds'])}
 jsonwrite(OUT/(stem+'_display_graph.json'),visual)
 with Image.open(OUT/(stem+'_source.jpg')) as source:photo=source.convert('RGB')
 assert photo.size==(record['image']['width'],record['image']['height'])
 canvas=Image.new('RGB',(1600,1200),'#F4F6FA');d=ImageDraw.Draw(canvas)
 d.rectangle((0,0,1600,104),fill='#152338');d.text((35,19),f'VG TEST  /  IMAGE {identity}',font=font(32,True),fill='white');d.text((35,63),'Context | seed 1729 | epoch 10 | model predictions',font=font(22),fill='#C8D7EA')
 view=fit(photo,(1536,976));ox=(1600-view.width)//2;oy=120+(976-view.height)//2;canvas.paste(view,(ox,oy));sx=view.width/photo.width;sy=view.height/photo.height
 labels=[]
 for n in node_records:
  x1,y1,x2,y2=n['box_xyxy'];x1=max(0,min(photo.width,x1));x2=max(x1,min(photo.width,x2));y1=max(0,min(photo.height,y1));y2=max(y1,min(photo.height,y2));box=(ox+x1*sx,oy+y1*sy,ox+x2*sx,oy+y2*sy)
  d.rectangle(box,outline=n['color'],width=5)
  text=f"{n['id']:02d}  {n['label']}";bb=d.textbbox((0,0),text,font=font(23,True));tw=bb[2]-bb[0]+18;th=38
  candidates=[(box[0],box[1]-th-2),(box[0],box[1]+2),(box[2]-tw,box[3]+2),(box[2]-tw,box[3]-th-2)]
  def place(c):
   x=max(ox,min(ox+view.width-tw,c[0]));y=max(oy,min(oy+view.height-th,c[1]));return (x,y,x+tw,y+th)
  def overlaps(a):return sum(max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1])) for b in labels)
  label_box=min(map(place,candidates),key=overlaps);labels.append(label_box);d.rounded_rectangle(label_box,radius=6,fill='white',outline=n['color'],width=3);d.text((label_box[0]+9,label_box[1]+2),text,font=font(23,True),fill='#152338')
 d.rectangle((0,1118,1600,1200),fill='white');d.text((32,1131),f'{len(nodes)} displayed object groups | {len(edges)} predicted relations | no ground-truth overlay',font=font(23),fill='#263B54');d.text((32,1163),'Boxes shown for the objects participating in the displayed graph.',font=font(20),fill='#5D6C7D')
 detection=OUT/(stem+'_detections.png');canvas.save(detection)
 lines=['digraph SceneGraph {','graph [rankdir=LR,bgcolor="white",pad="0.3",nodesep="0.48",ranksep="0.9",splines=spline,overlap=false,dpi=160];','node [shape=box,style="rounded,filled",fontname="Segoe UI",fontsize=20,margin="0.16,0.12",penwidth=2.4];','edge [fontname="Segoe UI",fontsize=16,color="#64748B",fontcolor="#263B54",arrowsize=0.85,penwidth=1.7];']
 for n in node_records:lines.append(f'n{n["id"]} [label={json.dumps(str(n["id"]).zfill(2)+" | "+n["label"])},color="{n["color"]}",fillcolor="{n["color"]}18",fontcolor="#152338"];')
 for e in edges:lines.append(f'n{e["subject_id"]} -> n{e["object_id"]} [label={json.dumps(e["predicate"])}];')
 lines.append('}');dot=OUT/(stem+'_scene_graph.dot');dot.write_text('\n'.join(lines),encoding='utf8')
 for fmt in ('png','svg'):
  subprocess.run([shutil.which('dot'),'-T'+fmt,str(dot),'-o',str(OUT/(stem+'_scene_graph.'+fmt))],check=True,capture_output=True,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
 combined=Image.new('RGB',(2200,1370),'#F4F6FA');cd=ImageDraw.Draw(combined);cd.rectangle((0,0,2200,115),fill='#152338');cd.text((42,22),f'CONTEXT  /  DETECTION + SCENE GRAPH  /  VG {identity}',font=font(34,True),fill='white');cd.text((42,71),'Trained checkpoint: seed 1729, epoch 10  |  held-out test sample  |  saved model outputs',font=font(24),fill='#C8D7EA')
 cd.text((36,144),'A   OBJECT DETECTIONS',font=font(25,True),fill='#263B54');cd.text((1280,144),'B   PREDICTED SCENE GRAPH',font=font(25,True),fill='#263B54')
 # Use the image+boxes region from the standalone detection figure.
 det_crop=canvas.crop((0,104,1600,1118));paste_center(combined,det_crop,(28,197,1210,865))
 with Image.open(OUT/(stem+'_scene_graph.png')) as graph:paste_center(combined,graph.convert('RGB'),(1260,197,912,865))
 cd.line((1250,190,1250,1070),fill='#D8E0EB',width=2)
 cd.text((40,1100),'DISPLAYED MODEL TRIPLETS',font=font(23,True),fill='#263B54')
 for i,e in enumerate(edges):
  col=i//6;row=i%6;text=f"{e['subject_id']:02d} {byid[e['subject_id']]['label']}  -- {e['predicate']} -->  {e['object_id']:02d} {byid[e['object_id']]['label']}"
  cd.text((40+col*1080,1140+row*30),text,font=font(20),fill='#34465E')
 cd.text((40,1332),'Qualitative subset only: up to12 relations /10 NMS groups. Predictions may be incorrect; no GT correction or new test scoring.',font=font(20),fill='#5D6C7D')
 combined.save(OUT/(stem+'_combined.png'))
 gallery.append({'image_id':identity,'stem':stem,'test_index':record['test_index'],'nodes':len(nodes),'relations':len(edges),'triplets':edges})

# Contact sheet uses the object panels for legibility; paired figures remain separate.
sheet=Image.new('RGB',(1800,1450),'#F4F6FA');sd=ImageDraw.Draw(sheet);sd.rectangle((0,0,1800,98),fill='#152338');sd.text((30,20),'CONTEXT 1729 / EPOCH 10  —  SIX FIXED VG TEST SAMPLES',font=font(29,True),fill='white');sd.text((30,62),'Detection previews; open individual paired figures for scene graphs.',font=font(21),fill='#C8D7EA')
for i,g in enumerate(gallery):
 with Image.open(OUT/(g['stem']+'_detections.png')) as im:paste_center(sheet,im.convert('RGB'),(i%2*900,110+(i//2)*443,900,438))
sheet.save(OUT/'contact_sheet.png')
cards=[]
for g in gallery:
 stem=g['stem'];cards.append(f'<section><h2>VG {g["image_id"]} <small>test index {g["test_index"]}</small></h2><a href="{stem}_combined.png"><img src="{stem}_combined.png" loading="lazy" alt="Predicted detections and scene graph for VG {g["image_id"]}"></a><p><a href="{stem}_source.jpg">Ảnh gốc</a> · <a href="{stem}_detections.png">Nhận diện</a> · <a href="{stem}_scene_graph.svg">Scene graph SVG</a> · <a href="{stem}_display_graph.json">SG JSON</a> · <a href="{stem}_prediction.json">Dự đoán gốc</a></p></section>')
page='''<!doctype html><html lang="vi"><meta charset="utf-8"><title>Context 1729 — VG test visualization</title><style>body{margin:0;background:#eef2f7;color:#152338;font:17px/1.6 'Segoe UI',sans-serif}main{max-width:1500px;margin:auto;padding:32px}h1{font-size:32px}header,section{background:white;padding:28px;margin-bottom:24px;border-radius:14px}img{width:100%;height:auto}small{font-weight:400;color:#64748b}a{color:#2457be}code{word-break:break-all}</style><main><header><h1>Nhận diện &amp; scene graph — Context seed 1729</h1><p>Checkpoint epoch 10. Sáu ảnh được chọn cố định bằng random seed 1729 từ VG test; không chọn theo chất lượng dự đoán.</p><p>Dùng lại dự đoán của lượt test đã hoàn tất, không chạy model hay chấm metric lại. Hình hiển thị tối đa 12 quan hệ đứng đầu giữa tối đa 10 nhóm đối tượng sau NMS; màu và số node khớp giữa hai phần. Đây là dự đoán của model, có thể sai; không dùng nhãn thật để sửa. Ảnh nhận diện chỉ hiển thị các đối tượng tham gia SG minh họa.</p><p>Model gồm detector VG-pretrained giữ cố định + relation head/Context đã train; đây là nhánh global Context, không phải minh chứng cho lợi ích của nhánh vùng.</p><p><a href="contact_sheet.png">Tổng quan 6 ảnh</a> · <a href="samples.json">Provenance và toàn bộ dự đoán mẫu</a></p></header>'''+''.join(cards)+'</main></html>'
(OUT/'index.html').write_text(page,encoding='utf8')
readme=f'''# Context seed1729 — qualitative VG test examples\n\nLocal output directory: `{OUT}`. Open `index.html` for6paired images and links to original JPEG, detections PNG, graph PNG/SVG/DOT and raw/display JSON.\n\nCheckpoint epoch10 SHA256 `{payload['checkpoint_sha256']}`. Saved TEST prediction gzip SHA256 `{payload['prediction_artifact_sha256']}` verified before sampling. Sampling indices: {payload['sample_indices']}, fixed random seed1729, without GT or correctness selection.\n\nNo new inference, evaluator or test metric was run. Original images were read directly from existing server ZIPs and downloaded only for these6authorized examples; no server extraction or image writes. The raw image JPEG hashes are in samples.json. Visualization is a display-only subset of the saved native-ranked predictions, up to12positive-ranking non-self NMS-group triplets/10groups. Object boxes use the highest-object-score query within each NMS group. Class/predicate labels are model predictions; no semantic editing or GT correction.\n\nThe current model is a frozen VG detector plus adapted relation/Context heads, not a fresh caption-clean detector or evidence for regional-method superiority. Boxes shown are objects participating in displayed SG, not all200detector queries. Raw object scores follow native softmax(sigmoid logits), are not calibrated probabilities, and are retained in JSON rather than labeled as certainty.\n'''
(OUT/'README.md').write_text(readme,encoding='utf8');jsonwrite(OUT/'gallery_manifest.json',gallery)
bundle=OUT.parent/(OUT.name+'.zip')
with zipfile.ZipFile(bundle,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(OUT.iterdir()):
  if p.is_file() and p.name!='remote_export.json':z.write(p,p.name)
print(json.dumps({'status':'RENDERED','directory':str(OUT),'gallery':str(OUT/'index.html'),'archive':str(bundle),'samples':len(gallery),'image_ids':[g['image_id'] for g in gallery],'png_count':len(list(OUT.glob('*.png'))),'new_inference':False,'new_test_scoring':False}))
