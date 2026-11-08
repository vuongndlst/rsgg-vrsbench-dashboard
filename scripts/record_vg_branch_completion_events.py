"""Local completion-only aggregation; no SSH, GPU, test access or job edits."""
from pathlib import Path
from datetime import datetime,timezone
import json

def complete_branches(snapshot):
    cfg=snapshot['active_config'];experiment=snapshot['active_experiment']['experiment_id']
    grouped={}
    for row in snapshot.get('runs',[]):
        stage=row['run'].split('_',1)[0]
        if stage not in ('screening','confirmation'):continue
        variant=row['variant'];seed=row['seed']
        assert row['run']==f'{stage}_{variant}_{seed}'
        bucket=grouped.setdefault((stage,variant),{})
        assert seed not in bucket,'duplicate_branch_seed'
        bucket[seed]=row
    completed=[]
    for (stage,variant),runs in sorted(grouped.items()):
        expected=[cfg['screening']['seed']] if stage=='screening' else cfg['confirmation']['seeds']
        assert set(runs).issubset(expected),'unregistered_seed'
        if set(runs)!=set(expected) or not all(row['status']=='complete' for row in runs.values()):continue
        rows=[]
        for seed in expected:
            row=runs[seed]
            assert row.get('experiment_id',experiment)==experiment
            assert len(row['history'])==cfg[stage]['epochs']
            assert 1<=row['selected_epoch']<=cfg[stage]['epochs']
            metric=row['selected_metrics']
            assert metric['mode']=='SGDet' and metric['constraint']=='GC' and metric['images']==5000
            digest=row['checkpoint_sha256']
            assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest)
            rows.append({'seed':seed,'run':row['run'],'selected_epoch':row['selected_epoch'],
                         'R@50':metric['R@50'],'mR@50':metric['mR@50'],
                         'guardrail_met':row['guardrail_met'],'checkpoint_sha256':digest})
        science_sha=snapshot['active_experiment'].get('scientific_preregistration_sha256',snapshot['active_experiment']['preregistration_sha256'])
        key='branch:'+experiment+':'+stage+':'+variant+':'+science_sha
        completed.append({'key':key,'type':'branch_complete','experiment_id':experiment,'stage':stage,
                          'variant':variant,'seeds':rows,
                          'mean_R@50':sum(row['R@50'] for row in rows)/len(rows),
                          'mean_mR@50':sum(row['mR@50'] for row in rows)/len(rows)})
    return completed

def new_events(completed,state,legacy_announced,initialize=False):
    announced=state.setdefault('announced',[]);events=[]
    for branch in completed:
        if branch['key'] in announced:continue
        old_keys=['checkpoint:'+row['run']+':'+row['checkpoint_sha256'] for row in branch['seeds']]
        historical=initialize and all(key in legacy_announced for key in old_keys)
        announced.append(branch['key'])
        if not historical:events.append(branch)
    return events

def main():
    root=Path(__file__).absolute().parents[1]
    snapshot=json.loads((root/'reports/VG_V2_LAST_STATUS.json').read_text(encoding='utf8'))
    path=root/'reports/VG_BRANCH_NOTIFY_STATE.json';initialize=not path.exists()
    state=json.loads(path.read_text(encoding='utf8')) if path.exists() else {'announced':[]}
    legacy=json.loads((root/'reports/VG_V2_MONITOR_STATE.json').read_text(encoding='utf8'))['announced']
    events=new_events(complete_branches(snapshot),state,legacy,initialize)
    out={'utc':datetime.now(timezone.utc).isoformat(),'snapshot_utc':snapshot['utc'],
         'notify':bool(events),'events':events,'additional_ssh_calls':0,
         'policy':'Whole screening branch, or all registered confirmation seeds; historical completed runs remain deduplicated'}
    def atomic(target,obj):
        temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(obj,indent=2),encoding='utf8');temporary.replace(target)
    atomic(path,state);atomic(root/'reports/VG_BRANCH_COMPLETION_LAST_CHECK.json',out)
    print(json.dumps(out))

if __name__=='__main__':main()
