"""Meaningful golden tests and numeric parity against pinned official NumPy metric classes."""
import sys,os,json,ast,time,unittest,hashlib
from pathlib import Path
import torch,numpy as np,h5py
from collections import defaultdict
from functools import reduce
from abc import ABC,abstractmethod
ROOT=Path.home()/'rsg-former';sys.path[:0]=[str(ROOT/'src'),str(ROOT/'.tools/deps')]
torch.set_num_threads(4)
from rsgformer.data import VG150
from rsgformer.geometry import directed_pairs,overlaps
from rsgformer.evaluation import PredClsEvaluator
from rsgformer.baselines import PairClassifier
runner=unittest.TextTestRunner(verbosity=2);suite=unittest.defaultTestLoader.discover(str(ROOT/'tests'))
result=runner.run(suite);assert result.wasSuccessful()
reference_root=ROOT/'research/references/vg_evaluator'
ns={'np':np,'torch':torch,'h5py':h5py,'defaultdict':defaultdict,'reduce':reduce,'ABC':ABC,'abstractmethod':abstractmethod,'FLIP_LEFT_RIGHT':0,'FLIP_TOP_BOTTOM':1}
def load(relative,names):
    path=reference_root/relative
    tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
load('maskrcnn_benchmark/structures/bounding_box.py',['BoxList'])
load('maskrcnn_benchmark/structures/boxlist_ops.py',['boxlist_iou'])
load('maskrcnn_benchmark/utils/miscellaneous.py',['bbox_overlaps','intersect_2d','argsort_desc'])
load('maskrcnn_benchmark/data/datasets/evaluation/vg/sgg_eval.py',['SceneGraphEvaluation','SGRecall','SGNoGraphConstraintRecall','SGMeanRecall','SGNGMeanRecall','_triplet','_compute_pred_matches'])
load('maskrcnn_benchmark/data/datasets/visual_genome.py',['load_graphs']);ns['BOX_SCALE']=1024
ds=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','val')
mask,b,c,attrs,rels=ns['load_graphs'](str(ROOT/'data/vg150/VG-SGG-with-attri.h5'),'val',-1,5000,True,False)
assert ds.indices==np.where(mask)[0].tolist()
loader_checks=[]
for i in [0,1,2,len(ds)//2,len(ds)-1]:
    s=ds[i];ref=ns['BoxList'](torch.from_numpy(b[i]/1024*max(s.width,s.height)),(s.width,s.height),'xyxy').clip_to_image(remove_empty=False)
    np.testing.assert_allclose(s.boxes,ref.bbox.numpy(),rtol=1e-6,atol=1e-4);np.testing.assert_array_equal(s.labels,c[i]);np.testing.assert_array_equal(s.relations,rels[i])
    loader_checks.append(s.image_id)
# Exact random-score metric parity on real validation annotations, with all 50 predicate classes.
rd={};gc=ns['SGRecall'](rd);ngc=ns['SGNoGraphConstraintRecall'](rd);mr=ns['SGMeanRecall'](rd,51,[str(i) for i in range(51)]);ngmr=ns['SGNGMeanRecall'](rd,51,[str(i) for i in range(51)])
for e in [gc,ngc,mr,ngmr]:e.register_container('predcls')
ours=PredClsEvaluator();rng=np.random.default_rng(1729);tested=0
cases=[('random',i) for i in range(16)]
trained=None
if (ROOT/'artifacts/A1_pilot.pt').exists():
    checkpoint_data=torch.load(ROOT/'artifacts/A1_pilot.pt',weights_only=True,map_location='cpu')
    trained=PairClassifier(True);trained.load_state_dict(checkpoint_data['model']);trained.eval()
    config_digest=hashlib.sha256((ROOT/'configs/p2_pilot.json').read_bytes()).hexdigest()[:12]
    cases += [('trained_A1',i) for i in range(16)]
for case,i in cases:
    s=ds[i];pairs=directed_pairs(len(s.labels));scores=rng.random((len(pairs),51));scores/=scores.sum(1,keepdims=True)
    if case=='trained_A1':
        feature=torch.load(ROOT/'artifacts'/('features_val_'+str(s.image_id)+'_'+config_digest+'.pt'),weights_only=True)
        with torch.no_grad():scores=trained(feature['objects'].float(),feature['unions'].float(),feature['labels'],feature['pairs'],feature['geometry']).softmax(-1).numpy()
    order=np.argsort(-scores[:,1:].max(1),kind='stable');pairs=pairs[order];scores=scores[order]
    if not len(pairs) or not len(s.relations):continue
    local={'pred_rel_inds':pairs,'rel_scores':scores,'gt_rels':s.relations,'gt_classes':s.labels,'gt_boxes':s.boxes,'pred_classes':s.labels,'pred_boxes':s.boxes,'obj_scores':np.ones(len(s.labels))}
    glob={'iou_thres':.5};local=gc.calculate_recall(glob,local,'predcls');ngc.calculate_recall(glob,local,'predcls');mr.collect_mean_recall_items(glob,local,'predcls');ngmr.collect_mean_recall_items(glob,local,'predcls')
    ours.update(s,pairs,scores);tested+=1
mr.calculate_mean_recall('predcls');ngmr.calculate_mean_recall('predcls');actual=ours.result();comparisons={}
for k in [20,50,100]:
    values={'gc_R':float(np.mean(rd['predcls_recall'][k])),'ngc_R':float(np.mean(rd['predcls_recall_nogc'][k])),'gc_mR':rd['predcls_mean_recall'][k],'ngc_mR':rd['predcls_ng_mean_recall'][k]}
    expected={'gc_R':actual['gc']['R@'+str(k)],'ngc_R':actual['ngc']['R@'+str(k)],'gc_mR':actual['gc']['mR@'+str(k)],'ngc_mR':actual['ngc']['mR@'+str(k)]}
    np.testing.assert_allclose(list(values.values()),list(expected.values()),rtol=0,atol=1e-12);comparisons[str(k)]=values
# Adapter train-index equality, including the native-scale overlap filter.
train=VG150(ROOT/'data/vg150',Path.home()/'sbir/hienlth/vg','train')
tmask,*_=ns['load_graphs'](str(ROOT/'data/vg150/VG-SGG-with-attri.h5'),'train',-1,5000,True,True)
assert train.indices==np.where(tmask)[0].tolist()
from rsgformer.geometry import pair_geometry
s=ds[0];p=directed_pairs(len(s.labels));p=p[:32];torch.manual_seed(1729)
m=PairClassifier(True);o=torch.randn(len(s.labels),2048);u=torch.randn(len(p),2048);labels=torch.tensor(s.labels);pt=torch.tensor(p);g=torch.tensor(pair_geometry(s.boxes,p,s.width,s.height))
m.eval();expected=m(o,u,labels,pt,g);checkpoint=ROOT/'artifacts/reload_test.pt';torch.save(m.state_dict(),checkpoint);n=PairClassifier(True);n.load_state_dict(torch.load(checkpoint,weights_only=True));n.eval();torch.testing.assert_close(expected,n(o,u,labels,pt,g),rtol=0,atol=0)
out={'status':'PASS','golden_tests':result.testsRun,'official_metric_parity_cases':tested,'official_metric_parity_unique_images':16,'prediction_sources':['random','trained_A1'] if trained else ['random'],'metric_parity_tolerance':1e-12,'loader_parity_image_ids':loader_checks,'train_split_indices_match_official':True,'checkpoint_reload_exact':True,'reference_commit':'ceb71fa88461c2a97a6258a80f47669d89207296','metric_values':comparisons,'intentional_deltas':['empty prediction counts zero instead of upstream silent skip','stable ordering for ties; parity scores have no score ties at cutoff'],'python_executable':sys.executable}
(ROOT/'reports/P2_TEST_REPORT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out),flush=True)
