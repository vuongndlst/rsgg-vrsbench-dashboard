import unittest
import numpy as np
from rsgformer.geometry import directed_pairs,overlaps,union_boxes,pair_geometry
from rsgformer.data import ImageGraphSample,ce_targets,VG150
from rsgformer.evaluation import PredClsEvaluator

def sample(rels):
    return ImageGraphSample(1,30,30,np.array([[0,0,9,9],[5,5,14,14],[20,20,29,29]],np.float32),np.array([1,2,3]),np.asarray(rels,np.int64).reshape(-1,3),np.array([101,102,103]))
class ProtocolTests(unittest.TestCase):
    def test_direction_and_union(self):
        p=directed_pairs(3);self.assertEqual(len(p),6);self.assertIn((1,0),map(tuple,p));self.assertIn((0,1),map(tuple,p))
        np.testing.assert_array_equal(union_boxes(sample([]).boxes,[[0,1],[1,0]]),[[0,0,14,14],[0,0,14,14]])
        g=pair_geometry(sample([]).boxes,np.array([[0,1],[1,0]]),30,30);np.testing.assert_allclose(g[0,8:10],-g[1,8:10])
    def test_inclusive_iou(self):
        self.assertAlmostEqual(float(overlaps([[0,0,0,0]],[[0,0,0,0]])[0,0]),1)
        self.assertAlmostEqual(float(overlaps([[0,0,9,9]],[[5,5,14,14]])[0,0]),25/175)
    def test_native_integer_box_conversion(self):
        d=VG150.__new__(VG150);d.arrays={'img_to_first_box':np.array([0]),'img_to_last_box':np.array([1]),'boxes_1024':np.array([[10,10,3,5],[0,0,3,3]],dtype=np.int32)}
        np.testing.assert_array_equal(d._native_boxes(0),[[8,7,11,12],[-1,-1,2,2]])
    def test_multilabel_constraints(self):
        s=sample([[0,1,1],[0,1,2]]);e=PredClsEvaluator(2);e.update(s,[[0,1]],[[0,.9,.8]])
        r=e.result();self.assertEqual(r['gc']['R@20'],.5);self.assertEqual(r['gc']['mR@20'],.5);self.assertEqual(r['ngc']['R@20'],1)
    def test_empty_predictions_count_zero(self):
        e=PredClsEvaluator(2);e.update(sample([[0,1,1]]),[[0,1]],[[0,.9,.1]])
        e.update(sample([[0,1,1]]),[],np.empty((0,3)))
        self.assertEqual(e.result()['gc']['R@50'],.5);self.assertEqual(e.result()['images_evaluated'],2)
    def test_wrong_predicate_and_wrong_direction(self):
        for p,scores in [([[1,0]],[[0,.9,.1]]),([[0,1]],[[0,.1,.9]])]:
            e=PredClsEvaluator(2);e.update(sample([[0,1,1]]),p,scores);self.assertEqual(e.result()['gc']['R@50'],0)
    def test_image_macro_recall(self):
        e=PredClsEvaluator(2);e.update(sample([[0,1,1]]),[[0,1]],[[0,.9,.1]])
        e.update(sample([[0,1,1],[1,0,1],[0,2,1]]),[[0,1]],[[0,.9,.1]])
        self.assertAlmostEqual(e.result()['gc']['R@50'],2/3);self.assertAlmostEqual(e.result()['gc']['mR@50'],1/3)
    def test_background_not_ranked(self):
        e=PredClsEvaluator(2);e.update(sample([[0,1,1]]),[[0,1]],[[100,.9,.1]])
        self.assertEqual(e.result()['gc']['R@50'],1)
    def test_duplicate_prediction_rejected(self):
        with self.assertRaises(AssertionError):PredClsEvaluator(2).update(sample([[0,1,1]]),[[0,1],[0,1]],[[0,.9,.1],[0,.8,.1]])
    def test_empty_graph_and_single_object(self):
        self.assertEqual(directed_pairs(1).shape,(0,2));e=PredClsEvaluator(2);e.update(sample([]),[],np.empty((0,3)))
        self.assertEqual(e.result()['images_evaluated'],0);self.assertIsNone(e.result()['gc']['R@50'])
    def test_train_duplicate_sampling_preserves_gt(self):
        s=sample([[0,1,1],[0,1,2]]);p=directed_pairs(3)
        a=ce_targets(s,p,np.random.default_rng(7));b=ce_targets(s,p,np.random.default_rng(7));np.testing.assert_array_equal(a,b)
        self.assertEqual(len(s.relations),2);self.assertEqual(int((a>0).sum()),1)
if __name__=='__main__':unittest.main()
