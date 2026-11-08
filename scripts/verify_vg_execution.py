"""Check current VG source/config lock and redact credentials without printing them."""
from pathlib import Path
import json,hashlib,re,shlex
ROOT=Path(__file__).absolute().parents[1]
def main():
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    locked=json.loads((ROOT/'reports/VG_METHOD_PREREGISTRATION.json').read_text())
    assert locked['config_sha256']==sha(ROOT/'configs/vg_method_v1.json')
    for relative,value in locked['sources'].items():assert sha(ROOT/relative)==value,relative
    tests=json.loads((ROOT/'reports/VG_METHOD_TESTS.json').read_text());probe=json.loads((ROOT/'reports/VG_METHOD_CONTRACT_PROBE.json').read_text())
    assert tests['status']=='PASS' and tests['tests']==7 and probe['status']=='PASS'
    assert not probe['validation_scored'] and not probe['test_scored']
    assert all(x['checkpoint_reload_exact'] for x in probe['variants'].values())
    for name,value in probe['source_checksums'].items():assert sha(ROOT/'src/rsgformer'/name)==value
    env={m.group(1):m.group(2).strip().strip('"').strip("'") for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$',(ROOT/'.env').read_text(),re.M)}
    secrets=[v for v in env.values() if len(v)>=6]
    for part in shlex.split(env.get('SSH_SERVER','')):
        if '@' in part and not part.startswith('-'):
            account,host=part.rsplit('@',1);secrets.extend([account,host])
    checked=syntax=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.tools' in path.parts or path.suffix not in ['.json','.jsonl','.md','.py','.csv','.txt','.bib','.html']:continue
        content=path.read_text(encoding='utf-8',errors='replace');assert not any(s and s in content for s in secrets),'redaction_check_failed';checked+=1
        if path.suffix=='.py' and (path.parent==ROOT/'scripts' or path.parent==ROOT/'tests' or 'src' in path.parts):compile(content,str(path),'exec');syntax+=1
    result={'status':'PASS','locked_source_files':len(locked['sources']),'config_matches_remote_preregistration':True,'mechanism_tests':7,'optimizer_checkpoint_probes':len(probe['variants']),'syntax_files':syntax,'text_files_redaction_checked':checked,'secret_leaks':0}
    (ROOT/'reports/VG_METHOD_LOCAL_INTEGRITY.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':main()
