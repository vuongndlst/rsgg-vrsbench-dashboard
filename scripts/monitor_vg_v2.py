"""Read-only SSH monitor, persistent event dedup; all experiment edits are separate."""
from pathlib import Path
import json,subprocess,sys,time,datetime
ROOT=Path(__file__).absolute().parents[1];STATE=ROOT/'reports/VG_V2_MONITOR_STATE.json';LAST=ROOT/'reports/VG_V2_MONITOR_LAST_CHECK.json'
def measured_remaining_seconds(current,progress,total_epochs):
 durations=[float(entry['seconds']) for entry in current.get('history',[])[-3:] if entry.get('seconds',0)>0]
 if not durations:return None
 duration=sum(durations)/len(durations);completed=len(current['history']);elapsed=0.
 # History seconds includes training and validation. Only subtract observed
 # training work in the following epoch; never subtract a completed epoch twice.
 if (progress.get('stage')=='training' and progress.get('run')==current['run']
     and progress.get('epoch')==completed+1):
  images=float(progress.get('images',0));rate=float(progress.get('step_seconds',0))
  if images>=0 and rate>0:elapsed=images*rate
 remaining=max(0.,max(0,total_epochs-completed)*duration-elapsed)
 return {'remaining_seconds':remaining,'mean_full_epoch_seconds':duration,
         'observed_current_training_seconds_subtracted':elapsed}

def save(path,obj):
 t=path.with_suffix('.tmp');t.write_text(json.dumps(obj,indent=2,ensure_ascii=False),encoding='utf-8');t.replace(path)
def main():
 state=json.loads(STATE.read_text()) if STATE.exists() else {'announced':[],'anomalies':[]};now=time.time();events=[]
 try:
  p=subprocess.run([sys.executable,'-X','utf8',str(ROOT/'scripts/ssh_project.py'),'run','scripts/remote_vg_v2_status.py'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=120)
  if p.returncode:raise RuntimeError('ssh_read_failed')
  status=json.loads(p.stdout);save(ROOT/'reports/VG_V2_LAST_STATUS.json',status)
 except Exception as e:
  condition='ssh_read_failed'
  if condition not in state['anomalies']:events.append({'type':'anomaly','condition':condition,'message':'Không đọc được SSH; chưa xác định job đã dừng.'})
  state['anomalies']=[condition];save(STATE,state);out={'notify':bool(events),'status':'monitor_read_failed','events':events};save(LAST,out);print(json.dumps(out,ensure_ascii=False));return
 def emit(key,event):
  if key not in state['announced']:events.append(event);state['announced'].append(key)
 readiness=status.get('VG_REGION_REFERENCE_PREPARATION.json',{})
 if readiness.get('status') in ['complete','failed']:
  fingerprint=readiness.get('json_sha256') or readiness.get('updated_utc','')
  emit('region_references:'+readiness['status']+':'+fingerprint,{'type':'reference_preparation_complete' if readiness['status']=='complete' else 'anomaly','stage':'vg_region_reference_preparation','status':readiness['status'],'image_records':readiness.get('image_records'),'region_count':readiness.get('region_count'),'reason':readiness.get('reason'),'caption_evaluation_performed':False})
 canonical=status.get('VG_CANONICAL_ANNOTATION_PREPARATION.json',{})
 if canonical.get('status')=='failed':
  emit('canonical_annotation_failure:'+canonical.get('updated_utc',''),{'type':'anomaly','stage':'canonical_annotation_preparation','reason':canonical.get('reason'),'active_v2_release_modified':False})
 parity=status.get('VG_CANONICAL_PARITY_AUDIT.json',{})
 if parity.get('status')=='metadata_audit_complete':
  emit('canonical_parity_audit:'+parity.get('archive_sha256',''),{'type':'metadata_audit_complete','stage':'canonical_vg_release_parity','strict_release_parity':parity.get('strict_release_parity'),'report':'reports/VG_CANONICAL_PARITY_AUDIT.json','test_performance_scored':False})
 oi=status.get('OPENIMAGES_TEST_METADATA.json',{})
 if oi.get('status')=='complete':
  emit('oi_test_metadata:'+oi.get('reference_sha256',''),{'type':'metadata_audit_complete','stage':'openimages_test_metadata','caption_images':oi.get('caption_images'),'human_captions':oi.get('human_captions'),'full_test_pixels_ready':False,'test_scored':False})
 pixels=status.get('OPENIMAGES_TEST_PIXEL_PILOT.json',{})
 if pixels.get('status')=='complete' and pixels.get('projected_storage_gate_pass') is False:
  emit('oi_test_storage_gate:128:12773:failed',{'type':'resource_gate_failed','stage':'openimages_full_pixel_cache','projected_safe_bytes':pixels.get('safety_projection_bytes'),'available_after_v2_and_reserve_bytes':pixels.get('additional_image_storage_with_v2_budget_and_reserve_bytes'),'full_test_download_launched':False,'other_program_work_continues':True})
 internal=status.get('VG_CROSS_SPLIT_PIXEL_AUDIT.json',{})
 if internal.get('status')=='CROSS_SPLIT_PIXEL_AUDIT_COMPLETE' and internal.get('groups_checked',0)>0:
  pairs=internal.get('split_pair_summary',{})
  emit('vg_cross_split_identity_overlap:'+internal.get('identity_source_sha256',''),{'type':'anomaly','condition':'internal_vg_split_photo_identity_overlap','metadata_groups':internal.get('groups_checked'),'near_pixel_groups_diagnostic':sum(x.get('phash_le4_rgb_mae_le3_groups',0) for x in pairs.values()),'split_pair_summary':pairs,'new_draft_exclusions':status.get('CAPTION_INTERNAL_IDENTITY_MERGE.json',{}).get('new_internal_exclusion_counts'),'active_v2_modified':False,'test_scored':False})
 for run in status.get('runs',[]):
  if run.get('status')=='complete':emit('checkpoint:'+run['run']+':'+run.get('checkpoint_sha256',''),{'type':'checkpoint_complete','experiment_id':run.get('experiment_id','vg_method_v2'),'run':run['run'],'selected_epoch':run['selected_epoch'],'metrics':run.get('selected_metrics'), 'completed_unix':run.get('completed_unix')})
 fresh_probe=status.get('FRESH_DETECTOR_NATIVE_GPU_PROBE.json',{})
 if fresh_probe.get('status')=='RESOURCE_OR_COMPATIBILITY_PROBE_FAILED':
  emit('fresh_native_gpu_probe_failure:'+fresh_probe.get('preregistration_sha256',''),
       {'type':'resource_gate_failed','stage':'fresh_detector_native_resolution_smoke',
        'cuda_oom':fresh_probe.get('cuda_oom',False),'allocator_cap_gib':fresh_probe.get('protocol',{}).get('resource',{}).get('cuda_allocator_gib'),
        'dataset_training_launched':False,'checkpoint_saved':False,'live_v2_modified':False,
        'test_scored':False,'report':'reports/FRESH_DETECTOR_NATIVE_GPU_PROBE.json',
        'next':'Preserve failed probe; independent activation-checkpoint compatibility preparation, then a separately admitted/versioned GPU pilot; never raise caps or change live v2'})
 for exact_report in ['CAPTION_EXACT_PIXEL_COMPARISON.json','CAPTION_EXACT_PIXEL_COMPARISON_V2.json','CAPTION_EXACT_PIXEL_COMPARISON_V3.json']:
  exact_pixels=status.get(exact_report,{})
  for split,identities in exact_pixels.get('retained_draft_ids_with_exact_protected_matches',{}).items():
   for identity in identities:
    emit('exact_retained_draft_protected_match:'+split+':'+str(identity),
         {'type':'anomaly','condition':'exact_protected_pixels_retained_in_provisional_draft',
          'canonical_split':split,'vg_id':identity,'evidence_sha256':exact_pixels.get('evidence_artifact_sha256'),
          'next':'Mandatory exclusion in a new final clean release before fresh training; retain old draft and live v2 unchanged',
          'active_v2_modified':False,'test_scored':False})
 large_headers=status.get('OI_FAILED_HEADER_PROBE_V3.json',{})
 if large_headers.get('status')=='HEADER_AUDIT_COMPLETE':
  blocked=[{'image_id':row['image_id'],'dimensions':row['header_dimensions']} for row in large_headers.get('records',[])
           if row.get('header_dimensions') and row['header_dimensions'][0]*row['header_dimensions'][1]>16000000]
  if blocked:
   emit('oi_source_primary_pixel_cap:reader_v2:16000000',
        {'type':'resource_gate_failed','condition':'oi_sources_exceed_registered_primary_frame_pixel_cap',
         'stage':'openimages_full_test_pixel_identity_audit','primary_frame_pixel_cap':16000000,'blocked_sources':blocked,
         'header_preregistration_sha256':large_headers['preregistration_sha256'],'full_body_retry_launched':False,
         'test_scored':False,'live_v2_modified':False,
         'next':'Preserve IDs and failures; current reader cannot certify these sources within its cap. Continue independent work; any new decoder/resource admission requires a separate locked version, no silent cap increase or missing-ID omission'})
 result=status.get('VG_V2_RESULT.json',{});terminal=result.get('status')
 oi_launch=status.get('OI_STREAM_PHASE_LAUNCH.json',{})
 oi_result=status.get('OI_STREAM_PHASE_RESULT.json',{})
 oi_audit=status.get('oi_stream_audit',{})
 exhausted=status.get('OI_STREAM_FINGERPRINT_PROGRESS.json',{}).get('retry_exhausted_images',0)
 if oi_launch.get('status') and exhausted>0:
  emit('oi_full_stream_retry_exhausted:'+oi_launch.get('preregistration_sha256',''),
       {'type':'anomaly','condition':'oi_test_pixel_sources_exhausted_registered_attempts',
        'stage':'openimages_full_test_pixel_identity_audit','missing_ids_at_first_notification':exhausted,
        'compatibility_evidence_report':'reports/OI_FAILED_HEADER_PROBE_V2.json',
        'supplemental_decoder_pilot':status.get('OI_MPO_REPAIR_PILOT.json',{}).get('status'),
        'next':'Preserve IDs/denominator and original phase limits; separately version compatibility repair/union, never claim complete source coverage from original journal',
        'test_scored':False,'live_v2_modified':False})
 if oi_result.get('status'):
  phase_sha=oi_result.get('preregistration_sha256','')
  emit('oi_full_stream_terminal:'+phase_sha+':'+oi_result['status'],
       {'type':'pixel_audit_stage_finished' if oi_result['status']=='pixel_index_complete' else 'anomaly',
        'stage':'openimages_full_test_pixel_identity_audit','outcome':oi_result['status'],
        'reason':oi_result.get('reason'),
        'indexed_images':(oi_result.get('last_progress') or {}).get('indexed_images'),
        'target_images':125436,'test_scored':False,'clean_release_certified':False,
        'next':'Join protected snapshots and audit perceptual candidates; preserve incomplete IDs and budget/provenance evidence'})
 elif oi_launch.get('status'):
  phase_sha=oi_launch.get('preregistration_sha256','')
  if not oi_audit.get('active'):
   emit('oi_full_stream_unexpected_exit:'+phase_sha,
        {'type':'anomaly','condition':'oi_identity_audit_process_exit_without_terminal_report',
         'stage':'openimages_full_test_pixel_identity_audit','vg_job_status_separate':True})
  else:
   oi_progress=oi_audit.get('progress',{})
   oi_signature=[oi_progress.get(k) for k in ['batches','indexed_images','retry_exhausted_images','persistent_network_charge_bytes']]
   if oi_signature!=state.get('oi_audit_progress_signature'):
    state['oi_audit_last_changed_unix']=now
   if now-state.get('oi_audit_last_changed_unix',now)>1800:
    emit('oi_full_stream_no_progress:'+phase_sha+':'+str(oi_progress.get('batches')),
         {'type':'anomaly','condition':'oi_identity_audit_no_progress_for30minutes',
          'stage':'openimages_full_test_pixel_identity_audit','vg_job_status_separate':True})
   state['oi_audit_progress_signature']=oi_signature
 experiment=status.get('active_experiment',{}).get('experiment_id','vg_method_v2')
 result_key=('result:' if experiment=='vg_method_v2' else 'result:'+experiment+':')
 if terminal:emit(result_key+terminal,{'type':'sgdet_stage_finished' if terminal!='stopped_on_error' else 'anomaly','experiment_id':experiment,'outcome':terminal,'reason':result.get('reason'),'test_scored':result.get('test_scored',False),'caption_stage':result.get('caption_stage')})
 progress=status.get('VG_V2_PROGRESS.json',{});active=any(p.get('active') for p in status['processes'].values());conditions=[]
 stamp=max([x.get('mtime',0) for x in status.get('downloads',[])]+[datetime.datetime.fromisoformat(progress['utc']).timestamp() if progress.get('utc') else 0])
 signature={'experiment_id':experiment,'stage':progress.get('stage'),'run':progress.get('run'),'epoch':progress.get('epoch'),'images':progress.get('images'),'download_bytes':sum(x.get('bytes',0) for x in status.get('downloads',[]))}
 if signature!=state.get('progress_signature'):state['last_changed_unix']=now
 if active and now-state.get('last_changed_unix',now)>1800:conditions.append('no_progress_for_30_minutes')
 if not active and not terminal:conditions.append('unexpected_process_exit')
 if status['disk_free_bytes']<16*1024**3:conditions.append('disk_reserve_below16GiB')
 for c in set(conditions)-set(state['anomalies']):events.append({'type':'anomaly','condition':c})
 eta={'status':'insufficient_measurements'}
 runs=[r for r in status.get('runs',[]) if r.get('history')];current=next((r for r in runs if r['run']==progress.get('run')),None)
 if current and not terminal:
  cfg=status.get('active_config') or json.loads((ROOT/'configs/vg_method_v2.json').read_text());phase=current['run'].split('_')[0];total=cfg[phase]['epochs'];measurement=measured_remaining_seconds(current,progress,total)
  if measurement:
   remaining=measurement['remaining_seconds']
   eta={'run':current['run'],'estimated_completion_Asia_Saigon':datetime.datetime.fromtimestamp(now+remaining,datetime.timezone(datetime.timedelta(hours=7))).strftime('%d/%m %H:%M'),'remaining_hours':round(remaining/3600,2),'method':'recent measured training+validation epoch durations minus observed current training work; provisional shared-GPU estimate',**measurement}
 elif progress.get('stage')=='feature_cache' and progress.get('estimated_remaining_seconds'):
  eta={'stage':'feature_cache','split':progress.get('split'),'estimated_completion_Asia_Saigon':datetime.datetime.fromtimestamp(now+progress['estimated_remaining_seconds'],datetime.timezone(datetime.timedelta(hours=7))).strftime('%d/%m %H:%M')}
 state.update(anomalies=conditions,last_check_unix=now,progress_signature=signature);save(STATE,state)
 out={'status':'checked','notify':bool(events),'events':events,'active':active,'sgdet_stage_terminal':bool(terminal),'outcome':terminal,'eta':eta,'caption_program_complete':False};save(LAST,out);print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
