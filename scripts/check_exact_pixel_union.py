"""Synthetic journal-row union contracts; no real data or network."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json
root=Path(__file__).absolute().parents[1]
source=root/'scripts/remote_caption_exact_pixel_compare_v2.py'
tree=ast.parse(source.read_text(encoding='utf8'))
functions=[node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in ['merge_protected_rows','protected_internal']]
assert len(functions)==2
namespace={}
exec(compile(ast.Module(body=functions,type_ignores=[]),'exact_union_fixture','exec'),namespace)
merge=namespace['merge_protected_rows']
first=('openimages_test','test','fake1','a'*64,'b'*64,5,7)
second=('openimages_test','test','fake2','c'*64,'d'*64,7,5)
rows,duplicates=merge([first],[first,second])
assert rows==[first,second] and duplicates==1
assert merge([first],[])[0]==[first]
try:
    merge([first],[first[:4]+('e'*64,5,7)])
except AssertionError as error:
    assert str(error)=='protected_identity_fingerprint_conflict'
else:
    raise AssertionError('conflicting_fingerprints_silently_coalesced')
assert not namespace['protected_internal']('test','train')
assert namespace['protected_internal']('train','test')
report={'utc':datetime.now(timezone.utc).isoformat(),'status':'PASS',
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'cases':['duplicate identity coalesced once','empty stream preserves local rows','conflicting fingerprint refuses','heldout direction'],
        'journal_modified':False,'benchmark_test_scored':False,'network_downloaded_bytes':0}
(root/'reports/CAPTION_EXACT_UNION_CONTRACT_PROBE.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
