import unittest
import numpy as np
import torch
from rsgformer.regions import RegionReasoner
from rsgformer.vg_experiment import collate_graphs,cache_pairs
from rsgformer.data import ImageGraphSample
from rsgformer.geometry import directed_pairs

def graph(n=4):
    torch.manual_seed(341+n);pairs=torch.from_numpy(directed_pairs(n));p=len(pairs)
    b=torch.rand(n,4);b[:,2:]=b[:,:2]+.1;b.clamp_(0,1)
    return {'image_id':42,'objects':torch.randn(n,2048),'unions':torch.randn(p,2048),'labels':torch.arange(n)+1,'pairs':pairs,'geometry':torch.randn(p,12),'boxes':b,'frequency_bias':torch.zeros(p,51)}

class RegionTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_valid_geometry_and_finite_memberships(self):
        m=RegionReasoner();a,_=collate_graphs([graph()]);logits,d=m(**a,diagnostics=True)
        self.assertTrue(torch.isfinite(logits).all());b=d['region_boxes']
        self.assertTrue((b>=0).all() and (b<=1).all() and (b[...,2:]>=b[...,:2]).all())
        self.assertTrue((d['alpha']>=0).all() and (d['alpha']<=1).all())
        self.assertGreater(float((b-d['initial_region_boxes']).abs().sum()),0)
    def test_geometry_feedback_gradient_without_feature_delta_path(self):
        torch.manual_seed(918);m=RegionReasoner();m.delta_to_feature.weight.data.zero_();a,_=collate_graphs([graph()])
        logits=m(**a);torch.nn.functional.cross_entropy(logits.flatten(0,1),torch.arange(logits.shape[1])%50+1).backward()
        self.assertGreater(float(m.geometry_update[-1].weight.grad.abs().sum()),1e-9)
        self.assertGreater(float(m.evidence[0].weight.grad.abs().sum()),1e-9)
    def test_capacity_control_identical_parameters_and_frozen_dynamic_geometry(self):
        torch.manual_seed(9);full=RegionReasoner('geometry_feedback');control=RegionReasoner('feature_membership');control.load_state_dict(full.state_dict());a,_=collate_graphs([graph()])
        y,dy=full(**a,diagnostics=True);z,dz=control(**a,diagnostics=True)
        self.assertEqual(sum(p.numel() for p in full.parameters()),sum(p.numel() for p in control.parameters()))
        torch.testing.assert_close(dz['region_boxes'],dz['initial_region_boxes'],rtol=0,atol=0)
        self.assertGreater(float((y-z).abs().sum()),0)
    def test_batch_padding_does_not_change_predictions(self):
        for variant in ['one_way','geometry_feedback','gnn']:
            m=RegionReasoner(variant).eval();f=graph(3);a,_=collate_graphs([f]);b,_=collate_graphs([f,graph(6)])
            with torch.no_grad():torch.testing.assert_close(m(**a)[0],m(**b)[0,:len(f['pairs'])],rtol=1e-5,atol=1e-6)
    def test_object_permutation_equivariance(self):
        m=RegionReasoner().eval();f=graph();a,_=collate_graphs([f]);order=torch.tensor([3,1,0,2]);inverse=torch.argsort(order);g=dict(f)
        for key in ['objects','labels','boxes']:g[key]=g[key][order]
        g['pairs']=inverse[g['pairs']];b,_=collate_graphs([g])
        with torch.no_grad():torch.testing.assert_close(m(**a),m(**b),rtol=1e-5,atol=1e-6)
    def test_candidate_cache_preserves_positives_deterministically(self):
        sample=ImageGraphSample(4,100,100,np.zeros((8,4),np.float32),np.arange(8)+1,np.array([[1,2,3],[1,2,4],[6,0,5]]),np.arange(8))
        a=cache_pairs(sample,7);np.testing.assert_array_equal(a,cache_pairs(sample,7));self.assertEqual(len(a),9)
        self.assertTrue({(1,2),(6,0)}<=set(map(tuple,a)))
    def test_no_gt_relation_argument_and_directed_inputs(self):
        m=RegionReasoner();a,_=collate_graphs([graph()]);y=m(**a)
        self.assertGreater(float((y[:,0]-y[:,3]).abs().sum()),0)
        with self.assertRaises(TypeError):m(**a,gt_relations=torch.zeros(1,3))
if __name__=='__main__':unittest.main()
