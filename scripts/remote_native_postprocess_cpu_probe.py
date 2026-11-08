"""Compare separate postprocessor with source-extracted native VG block on CPU."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from pathlib import Path
from datetime import datetime, timezone
import ast
import hashlib
import json
import signal
import sys
import textwrap
root = Path.home() / 'rsg-former'
sys.path[:0] = [str(root / 'src'), str(root / 'vendor/salience_sgg')]
signal.alarm(90)
import torch
import torchvision
import numpy as np
torch.set_num_threads(1)
from util import box_ops
from rsgformer.vg_native_postprocess import native_vg_postprocess
path = root / 'vendor/salience_sgg/models/salience_sggmodule.py'
source = path.read_text()
begin = source.index('            pred_rel = torch.clamp(')
end = source.index("            if 'oi_sgg_evaluator'", begin)
first = textwrap.dedent(source[begin:end])
begin = source.index('            demo_boxes = box_ops.box_cxcywh_to_xyxy(pred_boxes)')
end = source.index('            connections =', begin)
second = textwrap.dedent(source[begin:end])
prefix = '''def reference(out):
    j = 0
    pred_boxes = out['pred_boxes'][0]
    pred_logits = out['pred_logits'][0].sigmoid()
    obj_scores, pred_classes = pred_logits.softmax(-1).max(-1)
    pred_rel = out['pred_rel'][0].sigmoid()
    rel_confideces = out['pred_confidence'][-1].sigmoid()
'''
function = prefix + textwrap.indent(first + second, '    ') + '''
    return {'pred_rel_inds': demo[:100].numpy(), 'rel_scores': result_pred_rel_scores[:100].numpy(),
            'ranking_scores': triplet_scores[sorting_idx][keep_triplet == 1][:100].numpy(),
            'query_to_nms_equivalence_id': iou_assignments.numpy()}
'''
tree = ast.parse(function)
class DeviceOnly(ast.NodeTransformer):
    def visit_Constant(self, node):
        if node.value == 'cuda': return ast.copy_location(ast.Constant('cpu'), node)
        return node
tree = ast.fix_missing_locations(DeviceOnly().visit(tree))
namespace = {'torch':torch, 'torchvision':torchvision, 'box_ops':box_ops}
exec(compile(tree, '<pinned_native_postprocess_cpu_fixture>', 'exec'), namespace)
results = {}
for mode in ('random', 'overlapping_duplicate_queries', 'tied_logits'):
    torch.manual_seed(67182)
    n = 200
    out = {'pred_boxes':torch.rand(1,n,4)*.7+.1, 'pred_logits':torch.randn(1,n,150),
           'pred_rel':torch.randn(1,n,n,50), 'pred_confidence':torch.randn(1,1,n,n)}
    if mode == 'overlapping_duplicate_queries':
        out['pred_boxes'][0,1:8] = out['pred_boxes'][0,0]
        out['pred_logits'][0,1:8] = out['pred_logits'][0,0]
    if mode == 'tied_logits':
        out['pred_logits'].zero_(); out['pred_rel'].zero_(); out['pred_confidence'].zero_()
    expected = namespace['reference'](out)
    actual = native_vg_postprocess(out, 800, 600)
    for key in expected:
        assert np.array_equal(actual[key], expected[key]), (mode, key)
    assert np.all(actual['ranking_scores'][1:] <= actual['ranking_scores'][:-1])
    results[mode] = {'exact_query_pairs_predicate_scores_ranking_nms_ids': 'PASS',
                     'relation_rows': len(actual['pred_rel_inds']),
                     'native_zero_score_rows_retained': actual['native_zero_score_rows_retained']}
out = {'status':'PASS','utc':datetime.now(timezone.utc).isoformat(), 'synthetic_cases':results,
       'upstream_source_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
       'postprocessor_sha256':hashlib.sha256((root/'src/rsgformer/vg_native_postprocess.py').read_bytes()).hexdigest(),
       'reference_modification':'only CUDA device literals replaced with CPU in source-extracted fixture',
       'class_indexing':'zero-based', 'self_relation_scores':'zero before predicate argmax/dedup',
       'ranking':'object score product * salience * max predicate score; native Torch sort/dedup',
       'cuda_used':False, 'model_initialized':False, 'benchmark_validation_scored':False,
       'benchmark_test_scored':False, 'live_v2_modified':False,
       'remaining':['Full trained-model integration and clean-release/resource/preregistration gates',
                    'Caption graph evidence selection must be registered on independent dev; benchmark zero-score rows are not caption evidence']}
text=json.dumps(out,indent=2)
(root/'reports/NATIVE_POSTPROCESS_CPU_PROBE.json').write_text(text)
print(text)
