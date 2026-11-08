"""Synthetic notification/cadence fixture; no real state, jobs, network or scores."""
from pathlib import Path
import copy,hashlib,json,runpy
root=Path(__file__).absolute().parents[1]
branch_path=root/'scripts/record_vg_branch_completion_events.py'
module=runpy.run_path(str(branch_path),run_name='fixture_import')
cadence=runpy.run_path(str(root/'scripts/check_vg_heartbeat_cadence.py'),run_name='fixture_import')
base={'active_config':{'screening':{'seed':1729,'epochs':6},'confirmation':{'seeds':[1729,2718,3141],'epochs':13}},
      'active_experiment':{'experiment_id':'fixture','preregistration_sha256':'f'*64},'runs':[]}
def run(stage,seed,status='complete'):
    return {'run':f'{stage}_context_{seed}','variant':'context','seed':seed,'status':status,
            'history':[{} for _ in range(base['active_config'][stage]['epochs'])],
            'selected_epoch':1,'selected_metrics':{'mode':'SGDet','constraint':'GC','images':5000,'R@50':.25,'mR@50':.16},
            'guardrail_met':True,'checkpoint_sha256':format(seed,'064x')}
snapshot=copy.deepcopy(base);snapshot['runs']=[run('confirmation',1729),run('confirmation',2718)]
assert module['complete_branches'](snapshot)==[]
snapshot['runs'].append(run('confirmation',3141,'running'));assert module['complete_branches'](snapshot)==[]
snapshot['runs'][-1]['status']='complete';completed=module['complete_branches'](snapshot)
assert len(completed)==1 and len(completed[0]['seeds'])==3
state={'announced':[]};assert len(module['new_events'](completed,state,[]))==1
assert module['new_events'](completed,state,[])==[]
screen=copy.deepcopy(base);screen['runs']=[run('screening',1729)]
completed=module['complete_branches'](screen)
legacy=['checkpoint:screening_context_1729:'+format(1729,'064x')]
assert module['new_events'](completed,{'announced':[]},legacy,initialize=True)==[]
broken=copy.deepcopy(snapshot);broken['runs'].append(run('confirmation',3141))
try:module['complete_branches'](broken)
except AssertionError as exc:assert str(exc)=='duplicate_branch_seed'
else:raise AssertionError('duplicate_seed_accepted')
check=cadence['decision'];start='2026-10-12T03:24:00+00:00'
assert not check(start,'2026-10-12T04:24:00+00:00')['allow_remote_check']
assert check(start,'2026-10-12T15:24:00+00:00')['allow_remote_check']
assert check(start,'2026-10-12T04:24:00+00:00',direct=True)['allow_remote_check']
assert not check(start,'2026-10-12T02:24:00+00:00')['allow_remote_check']
report={'status':'PASS','contracts':['wait for all confirmation seeds','no partial running branch','single whole-branch event','durable dedup','historical bootstrap dedup','duplicate seed refusal','12hour cadence and direct-request exception'],
        'real_monitor_state_modified':False,'additional_ssh_calls':0,
        'source_sha256':hashlib.sha256(branch_path.read_bytes()).hexdigest()}
(root/'reports/VG_BRANCH_NOTIFICATION_POLICY_FIXTURE.json').write_text(json.dumps(report,indent=2),encoding='utf8')
print(json.dumps(report))
