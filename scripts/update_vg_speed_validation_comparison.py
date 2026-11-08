"""Record completed validation selections; never infer results for pending arms."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
root=Path(__file__).absolute().parents[1]
snapshot=json.loads((root/'reports/VG_V2_LAST_STATUS.json').read_text(encoding='utf8'))
experiment=snapshot['active_experiment']['experiment_id']
assert experiment=='vg_method_v2_speed_m2_v1'
lock_path=root/'reports/VG_V2_SPEED_PREREGISTRATION.json'
# ssh_project.fetch text transport uses CRLF on Windows; verify original LF bytes.
assert hashlib.sha256(lock_path.read_text(encoding='utf8').encode()).hexdigest()=='d5866565e34555143af0f2f42c240841eaa8e1f2b0152c05c4ef60121561377c'
runs={run['variant']:run for run in snapshot['runs'] if run['run'].startswith('screening_')}
reference=next(iter(runs.values()))['frozen_baseline_metrics']
assert reference['images']==5000 and reference['mode']=='SGDet' and reference['constraint']=='GC'
rows=[]
for variant in ('native','context','oneway','feedback'):
    run=runs.get(variant)
    row={'variant':variant,'seed':1729,'status':run['status'] if run else 'pending'}
    if run and run['status']=='complete':
        assert not run['test_scored'] and run['microbatch']==2
        assert run['frozen_baseline_metrics']==reference and len(run['history'])==6
        eligible=[h for h in run['history'] if h['metrics']['R@50']>=reference['R@50']-.01]
        selected=max(eligible or run['history'],key=lambda h:(h['metrics']['mR@50'],-h['epoch']))
        assert selected['epoch']==run['selected_epoch'] and selected['metrics']==run['selected_metrics']
        assert run['guardrail_met']==bool(eligible)
        metrics=run['selected_metrics']
        row.update(selected_epoch=run['selected_epoch'],validation_metrics=metrics,guardrail_met=run['guardrail_met'],
                   r50_percentage_point_change_vs_frozen=(metrics['R@50']-reference['R@50'])*100,
                   mr50_percentage_point_change_vs_frozen=(metrics['mR@50']-reference['mR@50'])*100,
                   checkpoint=run['checkpoint'],checkpoint_sha256=run['checkpoint_sha256'],
                   whole_run_wall_seconds=run['wall_seconds'],completed_unix=run['completed_unix'])
    rows.append(row)
out={'updated_utc':datetime.now(timezone.utc).isoformat(),'snapshot_utc':snapshot['utc'],
     'experiment_id':experiment,'preregistration_sha256':snapshot['active_experiment']['preregistration_sha256'],
     'protocol':'Validation5000 / SGDet GC / class and inclusive subject/object IoU>=0.5',
     'frozen_reference_metrics':reference,'screening_rows':rows,
     'all_four_arms_complete':all(row['status']=='complete' for row in rows),
     'confirmation_assessed_in_this_table':False,'screening_test_scored':False,'caption_clean_certified':False,
     'conclusion':'Completed screening selections only. Single-seed validation does not establish benefit from regions, superiority or caption-clean performance. Confirmation and locked test require the separately registered gates.'}
(root/'reports/VG_V2_SPEED_VALIDATION_COMPARISON.json').write_text(json.dumps(out,indent=2),encoding='utf8')
lines=['# VG optimized screening validation comparison','',
       'Snapshot: '+snapshot['utc']+'. '+out['protocol']+'. All scores below are percentages.','',
       '| Arm | Seed | Status | Selected epoch | R@50 | mR@50 |',
       '|---|---:|---|---:|---:|---:|',
       '| Frozen reference | — | fixed | — | %.4f | %.4f |'%(100*reference['R@50'],100*reference['mR@50'])]
for row in rows:
    if row['status']=='complete':
        m=row['validation_metrics'];lines.append('| %s |1729| complete |%d| %.4f | %.4f |'%(row['variant'],row['selected_epoch'],100*m['R@50'],100*m['mR@50']))
    else:lines.append('| %s |1729| %s | — | — | — |'%(row['variant'],row['status']))
lines+=['',out['conclusion'],'',
        'The superseded native run uses a different dropout RNG schedule and is historical only; it is not a matched control for this version. Fresh caption-clean detector training and caption evaluation require the remaining audited release and resource gates.']
(root/'reports/VG_V2_SPEED_VALIDATION_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
print(json.dumps({'comparison_saved':True,'completed_arms':[row['variant'] for row in rows if row['status']=='complete'],'test_scored':False}))
