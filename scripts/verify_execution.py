"""Verify current and historical manifests, split/config locks, syntax and redaction."""
from pathlib import Path
import hashlib,json,re,shlex
ROOT=Path(__file__).absolute().parents[1]

def main():
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    current=json.loads((ROOT/'research/P0_P1_P2_EVIDENCE_MANIFEST.json').read_text())
    old=json.loads((ROOT/'research/P0_EVIDENCE_MANIFEST.json').read_text())
    for f in current['files']:assert digest(ROOT/f['file'])==f['sha256'],f['file']
    for f in old['files']:assert digest(ROOT/'research'/f['file'])==f['sha256'],f['file']
    reference=json.loads((ROOT/'research/P0_EVALUATOR_LOCK.json').read_text())
    for f in reference['files']:assert digest(ROOT/'research/references/vg_evaluator'/f['path'])==f['sha256']
    state=json.loads((ROOT/'research/P0_P1_P2_STATUS.json').read_text())
    pilot=json.loads((ROOT/'reports/P2_PILOT_REPORT.json').read_text())
    prereg=json.loads((ROOT/'reports/P2_PILOT_PREREGISTRATION.json').read_text())
    assert pilot['config_sha256']==prereg['config_sha256']==digest(ROOT/'configs/p2_pilot.json')
    assert pilot['config']==prereg['config'] and not pilot['test_split_scored']
    assert pilot['gpu_hours_upper_bound']<20 and pilot['wall_seconds']<1800
    assert pilot['artifact_bytes']<15*1024**3 and pilot['peak_gpu_allocated_bytes']<8*1024**3
    assert state['evaluator']['status']=='PASS' and state['evaluator']['official_metric_parity_cases']==32
    assert all(pilot['runs'][k]['train_loss_decreased'] for k in ['A0','A1'])
    values={m.group(1):m.group(2).strip().strip('"').strip("'") for m in re.finditer(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)$',(ROOT/'.env').read_text(),re.M)}
    secrets=[v for v in values.values() if len(v)>=6]
    for part in shlex.split(values.get('SSH_SERVER','')):
        if '@' in part and not part.startswith('-'):
            account,host=part.rsplit('@',1);secrets.extend([account,host])
    leaks=[];checked=0;syntax=0
    for path in ROOT.rglob('*'):
        if not path.is_file() or '.tools' in path.parts or '.venv' in path.parts or path.suffix not in ['.py','.md','.json','.csv','.txt','.bib','.html','.jsonl']:continue
        content=path.read_text(encoding='utf-8',errors='replace');checked+=1
        if any(v and v in content for v in secrets):leaks.append(path.relative_to(ROOT).as_posix())
        if path.suffix=='.py' and path.parent.name!='vg_evaluator':
            # Compile only our code, not vendored research implementations.
            if 'src' in path.parts or 'scripts' in path.parts or 'tests' in path.parts or path.parent==ROOT/'research/tools':
                compile(content,str(path),'exec');syntax+=1
    assert not leaks,'credential_redaction_failed'
    output={'status':'PASS','current_manifest_files':len(current['files']),'historical_manifest_files':len(old['files']),'reference_source_files':len(reference['files']),'syntax_files':syntax,'text_files_redaction_checked':checked,'secret_leaks':0,'config_matches_preregistration':True,'resource_caps_met':True}
    (ROOT/'reports/LOCAL_INTEGRITY_CHECK.json').write_text(json.dumps(output,indent=2))
    print(json.dumps(output))
if __name__=='__main__':main()
