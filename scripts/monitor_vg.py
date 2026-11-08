"""Quiet event-based SSH monitoring with durable deduplication and measured checkpoint ETA."""
from pathlib import Path
from datetime import datetime,timezone,timedelta
import argparse,json,subprocess,sys,time,statistics,hashlib
ROOT=Path(__file__).absolute().parents[1];STATE=ROOT/'reports/VG_MONITOR_STATE.json'
ZONE=timezone(timedelta(hours=7))
def local_time(t):return datetime.fromtimestamp(t,ZONE).strftime('%d/%m %H:%M')
def save(state):
    p=STATE.with_suffix('.tmp');p.write_text(json.dumps(state,indent=2,ensure_ascii=False),encoding='utf-8');p.replace(STATE)
def estimate(status,config,probe,now):
    runs=status.get('screening',[])+status.get('confirmation',[]);completed={r['run'] for r in runs if r['status']=='complete'}
    progress=status.get('VG_METHOD_PROGRESS.json',{});current=progress.get('run')
    histories=[r for r in runs if r.get('epoch_seconds')]
    current_report=next((r for r in histories if r['run']==current),None)
    anchor=current_report or (histories[-1] if histories else None)
    if not anchor:return {'status':'insufficient_measured_epochs_for_ETA'}
    anchor_variant=next(v for v in config['screening']['variants'] if ('_'+v+'_') in anchor['run'])
    anchor_seconds=statistics.median(anchor['epoch_seconds'][-3:]);speed=probe['variants'][anchor_variant]['seconds_per_batch16']
    def duration(variant):
        observed=[s for r in histories if ('_'+variant+'_') in r['run'] for s in r['epoch_seconds'][-3:]]
        return statistics.median(observed) if observed else anchor_seconds*probe['variants'][variant]['seconds_per_batch16']/speed
    cursor=now;upcoming=[];screening_remaining=0;confirmation_remaining=0
    for phase in ['screening','confirmation']:
        settings=config[phase];seeds=[settings['seed']] if phase=='screening' else settings['seeds']
        for seed in seeds:
            for variant in settings['variants']:
                name=f'{phase}_{variant}_{seed}'
                if name in completed:continue
                if phase=='screening' and status.get('confirmation'):continue
                run=next((r for r in runs if r['run']==name),None);epochs=0 if run is None else run['epochs_finished'];epoch_seconds=duration(variant)
                seconds=max(0,settings['epochs']-epochs)*epoch_seconds
                if name==current and run and progress.get('epoch')==epochs+1:
                    seconds=max(0,epoch_seconds-max(0,now-run['report_updated_unix']))+max(0,settings['epochs']-epochs-1)*epoch_seconds
                cursor+=seconds
                upcoming.append({'run':name,'ETA_unix':cursor,'ETA_Asia_Saigon':local_time(cursor),'estimated_remaining_run_minutes':round(seconds/60,1),'conditional_on_screening_gate':phase=='confirmation' and not status.get('confirmation')})
                if phase=='screening':screening_remaining+=seconds
                else:confirmation_remaining+=seconds
    terminal=status.get('VG_METHOD_RESULT.json',{}).get('status')
    if terminal in ['screening_failed_to_establish_benefit','confirmation_did_not_meet_criteria','controlled_test_criteria_met','test_did_not_meet_criteria','stopped_on_error']:upcoming=[]
    return {'measured_epoch_seconds':round(anchor_seconds,1),'method':'recent full-data epoch medians; train-only architecture timing ratios for unmeasured variants; shared-GPU ETA is provisional','upcoming_checkpoints':upcoming,'screening_finish_Asia_Saigon':local_time(now+screening_remaining),'confirmation_finish_Asia_Saigon':local_time(now+screening_remaining+confirmation_remaining),'confirmation_conditional_on_gate':not bool(status.get('confirmation')),'locked_test_extra_hours_estimate':[1,3],'uncertainty':'allow approximately +/-25% remaining wall time; this is not a resource reservation'}
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--initialize',action='store_true');a=parser.parse_args();now=time.time()
    state=json.loads(STATE.read_text()) if STATE.exists() else {'announced_checkpoints':[],'announced_terminal':[],'active_anomalies':[]}
    events=[]
    try:
        p=subprocess.run([sys.executable,str(ROOT/'scripts/ssh_project.py'),'run','scripts/remote_vg_status.py','--environment','project'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
        if p.returncode:raise RuntimeError('ssh_status_read_failed')
        status=json.loads(p.stdout.strip());(ROOT/'reports/VG_METHOD_STATUS.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
    except Exception as exc:
        condition='monitor_connection_failure'
        if condition not in state.get('active_anomalies',[]):events.append({'type':'anomaly','condition':condition,'error_type':type(exc).__name__,'message':'Không đọc được SSH status; chưa xác định job đã dừng.'})
        state['active_anomalies']=list(set(state.get('active_anomalies',[])+[condition]));state['last_check_unix']=now;save(state)
        output={'status':'monitor_read_failed','notify':bool(events),'events':events}
        (ROOT/'reports/VG_MONITOR_LAST_CHECK.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(output,ensure_ascii=False));return
    config=json.loads((ROOT/'configs/vg_method_v1.json').read_text());probe=json.loads((ROOT/'reports/VG_METHOD_CONTRACT_PROBE.json').read_text());eta=estimate(status,config,probe,now)
    for r in status.get('screening',[])+status.get('confirmation',[]):
        if r['status']!='complete':continue
        key=r['run']+':'+r.get('checkpoint_sha256','unknown')
        if key not in state['announced_checkpoints']:
            if not a.initialize:events.append({'type':'checkpoint_complete','run':r['run'],'selected_epoch':r['selected_epoch'],'checkpoint':r.get('checkpoint'),'metrics':r.get('selected_metrics',{}),'completed_at_Asia_Saigon':local_time(r['report_updated_unix'])})
            state['announced_checkpoints'].append(key)
    result=status.get('VG_METHOD_RESULT.json',{});terminal=result.get('status')
    finished=terminal in ['screening_failed_to_establish_benefit','confirmation_did_not_meet_criteria','controlled_test_criteria_met','test_did_not_meet_criteria','stopped_on_error']
    if finished and terminal not in state['announced_terminal']:
        if not a.initialize:events.append({'type':'pipeline_finished' if terminal!='stopped_on_error' else 'anomaly','outcome':terminal,'reason':result.get('reason'),'test_scored':result.get('test_scored',False)})
        state['announced_terminal'].append(terminal)
    conditions=[]
    if not status.get('process_active',False) and not finished:conditions.append('unexpected_process_exit')
    progress=status.get('VG_METHOD_PROGRESS.json',{})
    if status.get('process_active') and progress.get('utc'):
        age=now-datetime.fromisoformat(progress['utc'].replace('Z','+00:00')).timestamp()
        if age>1800:conditions.append('no_progress_for_30_minutes')
    old_anomalies=set(state.get('active_anomalies',[]));new_anomalies=set(conditions)
    for condition in new_anomalies-old_anomalies:
        if not a.initialize:events.append({'type':'anomaly','condition':condition})
    for condition in old_anomalies-new_anomalies:
        if not a.initialize:events.append({'type':'anomaly_resolved','condition':condition})
    state.update(active_anomalies=conditions,last_check_unix=now,last_progress_utc=progress.get('utc'),last_process_active=status.get('process_active'),last_eta=eta,finished=finished)
    save(state);output={'status':'checked','notify':bool(events),'events':events,'finished':finished,'process_active':status.get('process_active'),'eta':eta,'initialization':a.initialize}
    (ROOT/'reports/VG_MONITOR_LAST_CHECK.json').write_text(json.dumps(output,indent=2,ensure_ascii=False),encoding='utf-8');print(json.dumps(output,ensure_ascii=False))
if __name__=='__main__':main()
