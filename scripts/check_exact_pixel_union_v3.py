"""Read-only evidence and invented-row contracts; no pixels or network."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json

root = Path(__file__).absolute().parents[1]
source = root / 'scripts/remote_caption_exact_pixel_compare_v3.py'
tree = ast.parse(source.read_text(encoding='utf8'))
functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
             and node.name in ('merge_protected_rows', 'protected_internal', 'supplemental_rows')]
assert len(functions) == 3
namespace = {'hashlib': hashlib, 'json': json}
exec(compile(ast.Module(body=functions, type_ignores=[]), 'union_v3_fixture', 'exec'), namespace)
merge = namespace['merge_protected_rows']
first = ('openimages_test', 'test', 'fake1', 'a'*64, 'b'*64, 5, 7)
second = ('openimages_test', 'test', 'fake2', 'c'*64, 'd'*64, 7, 5)
rows, duplicates = merge([first], [second], [first])
assert rows == [first, second] and duplicates == 1
assert merge([first], [], [])[0] == [first]
try:
    merge([first], [], [first[:4] + ('e'*64, 5, 7)])
except AssertionError as error:
    assert str(error) == 'protected_identity_fingerprint_conflict'
else:
    raise AssertionError('supplemental_conflict_not_rejected')
raw = (root / 'artifacts/caption_readiness/oi_mpo_repair_pilot_v1/pixel_fingerprints.json').read_bytes()
method = json.loads(raw)['protocol']['method']
supplemental, protocol = namespace['supplemental_rows'](raw, method)
assert len(supplemental) == 2
assert merge([], [], supplemental)[0] == sorted(supplemental)
for candidate, expected_method, reason in [(raw + b' ', method, 'supplemental_evidence_changed'),
                                          (raw, 'invented different method', 'supplemental_method_conflict')]:
    try:
        namespace['supplemental_rows'](candidate, expected_method)
    except AssertionError as error:
        assert str(error) == reason
    else:
        raise AssertionError('invalid_evidence_not_rejected')
assert not namespace['protected_internal']('test', 'train')
assert namespace['protected_internal']('train', 'test')
report = {'utc': datetime.now(timezone.utc).isoformat(), 'status': 'PASS',
          'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
          'cases': ['three-source identity dedup', 'empty source', 'supplemental conflict refusal',
                    'pinned actual evidence provenance', 'tamper refusal', 'method conflict refusal', 'heldout direction'],
          'journal_modified': False, 'benchmark_test_scored': False,
          'image_pixels_read_or_written': False, 'network_downloaded_bytes': 0}
(root / 'reports/CAPTION_EXACT_UNION_CONTRACT_PROBE_V3.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
