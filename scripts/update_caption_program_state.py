"""Record verified independent metadata preparation and preserve initial audit evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json

root = Path(__file__).absolute().parents[1]
snapshot = json.loads((root / 'reports/VG_V2_LAST_STATUS.json').read_text(encoding='utf8'))
prepared = snapshot['VG_REGION_REFERENCE_PREPARATION.json']
assert prepared['status'] == 'complete'
(root / 'reports/VG_REGION_REFERENCE_PREPARATION.json').write_text(json.dumps(prepared, indent=2), encoding='utf8')
initial = root / 'reports/VG_CAPTION_OVERLAP_AUDIT_INITIAL.json'
if not initial.exists():
    initial.write_text(json.dumps(snapshot['VG_CAPTION_OVERLAP_AUDIT.json'], indent=2), encoding='utf8')
lock = json.loads((root / 'reports/VG_V2_PREREGISTRATION.json').read_text(encoding='utf8'))
for name, expected in lock['source_hashes'].items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
path = root / 'reports/VG_PROGRAM_STATE.json'
state = json.loads(path.read_text(encoding='utf8'))
state['updated_utc'] = datetime.now(timezone.utc).isoformat()
state['caption_readiness'].update(reference_preparation_last_observed_status='complete',
                                  region_reference_image_records=prepared['image_records'],
                                  region_reference_count=prepared['region_count'],
                                  region_reference_sha256=prepared['json_sha256'],
                                  region_geometry_requires_preregistered_handling=prepared['invalid_geometry_records'])
active_experiment=snapshot.get('active_experiment',{}).get('experiment_id','vg_method_v2')
sgdet_state=state['sgdet_v2'] if active_experiment=='vg_method_v2' else state.setdefault('sgdet_optimized_version',{})
sgdet_state.update(last_observed_stage=snapshot['VG_V2_PROGRESS.json']['stage'],
                       last_snapshot_utc=snapshot['utc'], locked_local_source_hashes_verified=True)
if active_experiment!='vg_method_v2':
    new_lock=json.loads((root/'reports/VG_V2_SPEED_PREREGISTRATION.json').read_text(encoding='utf8'))
    for name,expected in new_lock['source_hashes'].items():
        candidate=root/name
        if candidate.exists():assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected,name
    execution_revision=snapshot.get('active_experiment',{}).get('execution_revision')
    if execution_revision:
        assert execution_revision=='vg_method_v2_time_extension_v1'
        extension=json.loads((root/'reports/VG_TIME_EXTENSION_V1_PREREGISTRATION.json').read_text(encoding='utf8'))
        assert extension['parent_preregistration_sha256']=='d5866565e34555143af0f2f42c240841eaa8e1f2b0152c05c4ef60121561377c'
        assert extension['config']==snapshot['active_config'] and not extension['scientific_protocol_changed']
        for name,expected in extension['source_hashes'].items():
            candidate=root/name
            if candidate.exists():assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected,name
        sgdet_state['execution_revision']=execution_revision
        sgdet_state['time_extension_preregistration']=extension
    state['sgdet_v2']['superseded_by']=active_experiment
    sgdet_state.update(experiment_id=active_experiment,config=snapshot.get('active_config',new_lock['config']),
                       all_four_arms_original_initialization=True,old_native_not_matched_control=True,
                       last_progress=snapshot['VG_V2_PROGRESS.json'],runs=snapshot.get('runs',[]),
                       test_scored=snapshot.get('VG_V2_RESULT.json',{}).get('test_scored',False))
epoch_durations = [entry['seconds'] for run in snapshot.get('runs', [])
                   for entry in run.get('history', []) if entry.get('seconds', 0) > 0]
if epoch_durations:
    cfg = snapshot.get('active_config') or lock['config']
    mean_epoch_hours = sum(epoch_durations) / len(epoch_durations) / 3600
    elapsed_hours = snapshot['VG_V2_PROGRESS.json']['wall_seconds'] / 3600
    screening_epochs = len(cfg['screening']['variants']) * cfg['screening']['epochs']
    completed_screen_epochs = sum(len(run.get('history', [])) for run in snapshot.get('runs', []) if run['run'].startswith('screening_'))
    completed_confirmation_epochs = sum(len(run.get('history', [])) for run in snapshot.get('runs', []) if run['run'].startswith('confirmation_'))
    remaining_screen_hours = max(0, screening_epochs - completed_screen_epochs) * mean_epoch_hours
    cap_hours = cfg['resource']['wall_hours']
    sgdet_state['provisional_budget_forecast'] = {
        'observed_completed_epochs': len(epoch_durations), 'mean_measured_epoch_hours': mean_epoch_hours,
        'assumption': 'future arms/confirmation epoch durations equal observed native epoch mean; shared GPU variability unresolved',
        'active_epoch_partial_work_not_subtracted': True,
        'wall_cap_hours': cap_hours,
        'observed_completed_screening_epochs': completed_screen_epochs,
        'observed_completed_confirmation_epochs': completed_confirmation_epochs,
        'remaining_screening_point_estimate_hours': remaining_screen_hours,
        'screening_only_projected_total_hours': elapsed_hours + remaining_screen_hours,
        'confirmation_paths': {
            name: {'planned_confirmation_epochs': arms * len(cfg['confirmation']['seeds']) * cfg['confirmation']['epochs'],
                   'projected_pipeline_total_hours': elapsed_hours + remaining_screen_hours + max(0, arms * len(cfg['confirmation']['seeds']) * cfg['confirmation']['epochs'] - completed_confirmation_epochs) * mean_epoch_hours,
                   'requires_budget_reassessment_if_screen_gate_passes': elapsed_hours + remaining_screen_hours + max(0, arms * len(cfg['confirmation']['seeds']) * cfg['confirmation']['epochs'] - completed_confirmation_epochs) * mean_epoch_hours > cap_hours}
            for name, arms in [('native_context_candidate', 3), ('feedback_with_oneway_control', 4)]},
        'excluded': ['test evaluation/bootstrap', 'fresh detector training', 'caption generation/evaluation'],
        'current_cap_exceeded': elapsed_hours > cap_hours, 'live_job_or_criteria_modified': False,
        'action': 'Reassess admission/resource authorization if screening passes; never silently raise cap or weaken confirmation gate'}
state['next_independent_actions'][0] = 'Freeze crop geometry/reference eligibility rules before caption evaluation; retain complete annotation source'
parity_path = root / 'reports/VG_CANONICAL_PARITY_AUDIT.json'
if parity_path.exists():
    parity = json.loads(parity_path.read_text(encoding='utf8'))
    state['canonical_release_audit'] = {
        'status': parity['status'], 'archive_sha256': parity['archive_sha256'],
        'strict_native_preprocessing_parity': parity['strict_release_parity'],
        'all_ordered_split_ids_equal': all(v['ordered_ids_equal'] for v in parity['splits'].values()),
        'all_unique_relation_sets_equal': all(v['unique_relation_set_mismatch_images'] == 0 for v in parity['splits'].values()),
        'active_v2_modified': False,
        'next_step': 'Build versioned canonical ZIP-backed annotation adapter and evaluator compatibility smoke before separate preregistration; retain existing v2 protocol/result'
    }
adapter_path = root / 'reports/VG_CANONICAL_ADAPTER_PROBE.json'
if adapter_path.exists():
    adapter = json.loads(adapter_path.read_text(encoding='utf8'))
    state['canonical_adapter_preparation'] = {
        'status': adapter['status'], 'images_checked': adapter['images_checked'],
        'adapter_sha256': adapter['adapter_sha256'],
        'train_sorted_ids_sha256': adapter['train_sorted_ids_sha256'],
        'cuda_used': False, 'new_training_launched': False,
        'clean_release_certified': False, 'remaining': adapter['remaining']
    }
    state['next_independent_actions'][1] = 'Canonical adapter target parity PASS; audit native augmentation/criterion/evaluator and fresh backbone initialization before separate preregistration'
oi_path = root / 'reports/OPENIMAGES_TEST_METADATA.json'
merge_path = root / 'reports/CAPTION_CLEAN_IDENTITY_MERGE.json'
if oi_path.exists() and merge_path.exists():
    oi = json.loads(oi_path.read_text(encoding='utf8'))
    merged = json.loads(merge_path.read_text(encoding='utf8'))
    assert oi['status'] == 'complete'
    state['openimages_test_readiness'] = {
        'scope': 'test_only_no_train_or_val', 'test_images': oi['test_images'],
        'human_captions': oi['human_captions'], 'vrd_non_attribute_images': oi['vrd_non_attribute_images'],
        'reference_sha256': oi['reference_sha256'], 'full_test_pixels_ready': False,
        'test_scored': False, 'next': 'bounded deterministic test-only pixel download pilot; then project storage before bulk download'
    }
    state['caption_readiness'].update(known_vg_train_val_external_dev_test_exclusions=merged['combined_excluded_train_val_images'],
                                     provisional_sorted_split_counts=merged['provisional_sorted_split_counts'],
                                     identity_merge_manifest_sha256=merged['manifest_sha256'])
pilot_path = root / 'reports/OPENIMAGES_TEST_PIXEL_PILOT_SUMMARY.json'
if pilot_path.exists():
    pilot = json.loads(pilot_path.read_text(encoding='utf8'))
    if pilot.get('status') == 'complete':
        state['openimages_test_readiness'].update(pixel_pilot_valid_images=pilot['valid_images'],
                                                  full_pixel_cache_storage_gate_pass=pilot['projected_storage_gate_pass'],
                                                  full_pixel_cache_launched=False,
                                                  next='Prepare bounded streaming/on-demand TEST reader and versioned evaluation protocol; no tuning or scoring before method/prompt locks')
init_path = root / 'reports/FRESH_BACKBONE_INITIALIZATION_AUDIT.json'
if init_path.exists():
    init = json.loads(init_path.read_text(encoding='utf8'))
    state['fresh_initialization_audit'] = {
        'status': init['status'], 'pretraining': init['pretraining'],
        'weights_sha256': init['weights_sha256'], 'vg_checkpoint_loaded': False,
        'fully_contamination_free_claim': False,
        'next': 'Fresh native detector build with ImageNet-only backbone and new query/class/box heads; recompute SGG prior from final clean train IDs'
    }
fresh_path = root / 'reports/FRESH_DETECTOR_CPU_PROBE.json'
if fresh_path.exists():
    fresh = json.loads(fresh_path.read_text(encoding='utf8'))
    assert fresh['status'] == 'PASS' and not fresh['vg_checkpoint_loaded']
    assert not fresh['cuda_used'] and not fresh['training_started']
    state['fresh_detector_preparation'] = {
        'status': fresh['status'], 'report_utc': fresh['utc'],
        'initialization': fresh['initialization'],
        'weights_sha256': fresh['weights_sha256'],
        'builder_sha256': fresh['source_hashes']['src/rsgformer/vg_fresh_detector.py'],
        'parameters': fresh['parameters'], 'trainable_parameters': fresh['trainable_parameters'],
        'synthetic_native_criterion_and_matching': 'PASS',
        'cuda_used': False, 'detector_forward_executed': False,
        'new_training_launched': False, 'clean_release_certified': False,
        'remaining': fresh['remaining']
    }
training_path = root / 'reports/CANONICAL_TRAINING_CPU_PROBE.json'
if training_path.exists():
    training = json.loads(training_path.read_text(encoding='utf8'))
    assert training['status'] == 'PASS' and not training['training_started']
    state['canonical_training_preparation'] = {
        'status': training['status'], 'report_utc': training['utc'],
        'scope': training['scope'], 'augmented_target_cases': len(training['cases']),
        'provisional_train_ids_sha256': training['provisional_train_ids_sha256'],
        'adapter_sha256': training['source_hashes']['src/rsgformer/vg_canonical_training.py'],
        'detection_has_native_random_crop': training['detection_has_native_random_crop'],
        'sgg_has_native_random_crop': training['sgg_has_native_random_crop'],
        'cuda_used': False, 'new_training_launched': False, 'clean_release_certified': False,
        'remaining': ['Final image overlap/exclusion audit', 'Full detector GPU forward/backward resource smoke',
                      'Separate training and selection preregistration', 'Canonical evaluator parity and clean-training SGG prior']
    }
reader_path = root / 'reports/OPENIMAGES_TEST_READER_PROBE.json'
if reader_path.exists():
    reader = json.loads(reader_path.read_text(encoding='utf8'))
    assert reader['status'] == 'PASS' and not reader['test_scored']
    state['openimages_test_readiness'].update(
        streaming_reader_local_pilot_contract='PASS',
        reader_sha256=reader['reader_sha256'],
        reader_probe_images=reader['existing_pilot_images_decoded'],
        full_pixel_coverage_audited=False, network_reader_branch_tested=False,
        next='Bounded network/completeness journal contracts; freeze method/prompt/decoder/orientation/evaluator before TEST inference')
prior_path = root / 'reports/CLEAN_TRAIN_PRIOR_PREPARATION.json'
if prior_path.exists():
    prior = json.loads(prior_path.read_text(encoding='utf8'))
    assert prior['status'] == 'PROVISIONAL_TRAIN_PRIOR_PREPARED' and not prior['used_in_any_model']
    state['clean_training_prior_preparation'] = {
        'status': prior['status'], 'report_utc': prior['utc'],
        'native_full_train_count_reconstruction_exact': prior['author_prior_exactly_equals_zero_based_counts'],
        'draft_prior_sha256': prior['draft_prior_sha256'],
        'provisional_train_images': prior['provisional_train_images'],
        'provisional_train_relations': prior['provisional_train_relations'],
        'object_indexing': prior['draft_prior_object_indexing'],
        'used_in_any_model': False, 'clean_release_certified': False,
        'remaining': prior['remaining']
    }
flickr_path = root / 'reports/FLICKR_PUBLIC_DOWNLOAD_PROBE.json'
if flickr_path.exists():
    flickr = json.loads(flickr_path.read_text(encoding='utf8'))
    state['flickr_source_readiness'] = {
        'public_box_file_id': flickr['public_file_id'],
        'official_archive_bytes': flickr['official_archive_bytes'],
        'public_http_download_verified': flickr['download_source_verified'],
        'last_http_status': flickr.get('http_status'),
        'missing_test_images': 966, 'new_images_downloaded': 0, 'test_scored': False,
        'next': 'Use the authorized official Box download flow or official image access; preserve missing-image gate, no unofficial/test-subset fallback'
    }
evaluator_path = root / 'reports/CANONICAL_EVALUATOR_CPU_PROBE.json'
if evaluator_path.exists():
    evaluator = json.loads(evaluator_path.read_text(encoding='utf8'))
    assert evaluator['status'] == 'PASS' and not evaluator['benchmark_test_scored']
    state['canonical_evaluator_preparation'] = {
        'status': evaluator['status'], 'report_utc': evaluator['utc'],
        'scope': evaluator['scope'],
        'adapter_sha256': evaluator['source_hashes']['src/rsgformer/vg_canonical_evaluation.py'],
        'iou_dtype': 'float64', 'iou_convention': 'pinned native inclusive +1',
        'synthetic_contract': 'PASS', 'compiled_extension_executed': False,
        'benchmark_validation_scored': False, 'benchmark_test_scored': False,
        'live_v2_modified': False, 'remaining': evaluator['remaining']
    }
postprocess_path = root / 'reports/NATIVE_POSTPROCESS_CPU_PROBE.json'
if postprocess_path.exists():
    postprocess = json.loads(postprocess_path.read_text(encoding='utf8'))
    assert postprocess['status'] == 'PASS' and not postprocess['benchmark_test_scored']
    state['native_postprocess_preparation'] = {
        'status': postprocess['status'], 'report_utc': postprocess['utc'],
        'postprocessor_sha256': postprocess['postprocessor_sha256'],
        'upstream_source_sha256': postprocess['upstream_source_sha256'],
        'synthetic_cases': postprocess['synthetic_cases'],
        'reference_modification': postprocess['reference_modification'],
        'benchmark_validation_scored': False, 'benchmark_test_scored': False,
        'live_v2_modified': False, 'remaining': postprocess['remaining']
    }
caption_runtime_path = root / 'reports/CAPTION_RUNTIME_PREFLIGHT.json'
if caption_runtime_path.exists():
    caption_runtime = json.loads(caption_runtime_path.read_text(encoding='utf8'))
    assert not caption_runtime['model_weights_loaded'] and not caption_runtime['benchmark_test_scored']
    state['caption_runtime_preparation'] = {
        'status': caption_runtime['status'], 'report_utc': caption_runtime['utc'],
        'versions': caption_runtime['versions'], 'java': caption_runtime['java'],
        'cpu_synthetic_scorers_pass': [name for name, value in caption_runtime['scorers'].items()
                                       if value.get('synthetic_finite_contract') == 'PASS'],
        'qwen_tokenizer': caption_runtime['qwen_tokenizer'],
        'full_pycocoevalcap_runtime_verified': False,
        'model_weights_loaded': False, 'benchmark_test_scored': False,
        'remaining': caption_runtime['remaining']
    }
java_preparation_path = root / 'reports/CAPTION_JAVA_PREPARATION.json'
if java_preparation_path.exists():
    java_preparation = json.loads(java_preparation_path.read_text(encoding='utf8'))
    assert java_preparation['status'] == 'PROJECT_LOCAL_JRE_PREPARED'
    assert not java_preparation['shared_environment_modified']
    state['caption_java_preparation'] = {
        'status': java_preparation['status'], 'report_utc': java_preparation['utc'],
        'release_name': java_preparation['release_name'],
        'archive_sha256': java_preparation['archive_sha256'],
        'archive_bytes': java_preparation['archive_bytes'],
        'unpacked_regular_file_bytes': java_preparation['unpacked_regular_file_bytes'],
        'java_path_alias': java_preparation['java_path_alias'],
        'java_executable_sha256': java_preparation['java_executable_sha256'],
        'shared_environment_modified': False, 'global_path_modified': False,
        'caption_benchmark_scored': False, 'remaining': java_preparation['remaining']
    }
metric_probe_path = root / 'reports/CAPTION_JAVA_METRIC_PROBE.json'
if metric_probe_path.exists():
    metric_probe = json.loads(metric_probe_path.read_text(encoding='utf8'))
    if metric_probe.get('status') == 'PASS':
        assert set(metric_probe['scorers']) == {'PTBTokenizer', 'BLEU', 'CIDEr', 'ROUGE_L', 'METEOR', 'SPICE'}
        assert all(value['status'] == 'PASS' for value in metric_probe['scorers'].values())
        assert not metric_probe['benchmark_test_scored'] and not metric_probe['shared_package_files_modified']
        state['caption_runtime_preparation'].pop('full_pycocoevalcap_runtime_verified', None)
        state['caption_runtime_preparation'].update(
            synthetic_cpu_runtime_verified=True, report_utc=metric_probe['utc'],
            synthetic_metric_contracts=metric_probe['scorers'],
            scorer_source_hashes=metric_probe['scorer_source_hashes'],
            java_runtime_used=metric_probe['java_release'],
            java8_launcher_adapter=metric_probe['java8_launcher_adapter'],
            fixture_script_sha256=hashlib.sha256((root / 'scripts/remote_caption_java_metric_probe.py').read_bytes()).hexdigest(),
            full_corpus_protocol_verified=False, benchmark_test_scored=False,
            remaining=['Freeze batch metric wrapper/runtime/source hashes and corpus tokenization/metric options',
                       'Frozen Qwen14B inference quantization/offload and aggregate resource pilot',
                       'Finalize clean detector/graph release and independent prompt selection',
                       'Locked caption evaluation and paired image confidence intervals'])
        state['caption_java_preparation']['remaining'] = ['Lock Java8 launcher adapter for corpus evaluation; keep shared/global environment unchanged']
state['next_independent_actions'][1] = 'Canonical data/augmentation/evaluator/postprocessor CPU contracts prepared; finalize external pixel overlap release, full detector resource smoke and separate training preregistration'
forward_probe_path = root / 'reports/FRESH_DETECTOR_FORWARD_CPU_PROBE.json'
if forward_probe_path.exists():
    forward_probe = json.loads(forward_probe_path.read_text(encoding='utf8'))
    assert forward_probe['status'] == 'PASS' and not forward_probe['vg_checkpoint_loaded']
    assert not forward_probe['dataset_training_started'] and not forward_probe['cuda_used']
    state['fresh_detector_preparation'].update(
        low_resolution_full_model_cpu_smoke='PASS',
        full_model_cpu_smoke_report_utc=forward_probe['utc'],
        detector_forward_executed=True, detector_backward_executed=True,
        smoke_image_shape=forward_probe['image_shape'],
        smoke_peak_cpu_rss_bytes=forward_probe['peak_rss_bytes'],
        finite_gradient_tensors=forward_probe['finite_gradient_tensors'],
        nonzero_gradient_tensors_by_group=forward_probe['nonzero_gradient_tensors_by_group'],
        optimizer_step_executed=False, new_training_launched=False,
        native_resolution_gpu_resource_gate_passed=False,
        remaining=forward_probe['remaining'])
internal_merge_path = root / 'reports/CAPTION_INTERNAL_IDENTITY_MERGE.json'
if internal_merge_path.exists():
    internal_merge = json.loads(internal_merge_path.read_text(encoding='utf8'))
    assert not internal_merge['clean_release_certified'] and not internal_merge['live_v2_modified']
    state['caption_internal_identity_protection'] = internal_merge
    state['caption_readiness'].update(current_provisional_identity_manifest_sha256=internal_merge['manifest_sha256'],
                                     current_provisional_sorted_split_counts=internal_merge['provisional_split_counts'],
                                     current_provisional_sorted_split_sha256=internal_merge['provisional_split_sha256'])
    state['clean_training_prior_preparation'].update(draft_matches_current_identity_manifest=False,
                                                    used_in_any_model=False,
                                                    regenerate_after_final_release=True)
    state['canonical_training_preparation']['probe_used_previous_provisional_identity_manifest'] = True
identity_inventory_path = root / 'reports/CAPTION_IDENTITY_SOURCE_INVENTORY.json'
if identity_inventory_path.exists():
    identity_inventory = json.loads(identity_inventory_path.read_text(encoding='utf8'))
    state['caption_identity_field_audit'] = {
        'metadata_sha256': identity_inventory['metadata_sha256'],
        'identity_field_coverage': identity_inventory['identity_field_coverage'],
        'missing_flickr_identity_counts_in_previous_manifest': identity_inventory['missing_flickr_identity_counts'],
        'fully_clean_release_certified': False
    }
coco_identity_path = root / 'reports/COCO_PROTECTED_IDENTITY_PREPARATION.json'
if coco_identity_path.exists():
    coco_identity = json.loads(coco_identity_path.read_text(encoding='utf8'))
    if coco_identity.get('status') == 'FULL_PROTECTED_COCO_METADATA_IDENTITIES_PREPARED':
        assert coco_identity['protected_coco_ids'] == coco_identity['protected_ids_with_flickr_identity'] == 10000
        assert not coco_identity['benchmark_test_scored'] and not coco_identity['training_manifest_modified']
        state['caption_coco_identity_preparation'] = coco_identity
oi_network_path = root / 'reports/OPENIMAGES_NETWORK_READER_PROBE.json'
if oi_network_path.exists():
    oi_network = json.loads(oi_network_path.read_text(encoding='utf8'))
    assert oi_network['status'] == 'PASS' and not oi_network['test_scored']
    assert not oi_network['reference_captions_read'] and oi_network['disk_pixel_bytes_written'] == 0
    state['openimages_test_readiness'].update(network_reader_branch_tested=True,
                                             network_probe_report_utc=oi_network['utc'],
                                             network_probe_images=oi_network['network_images_fully_decoded'],
                                             network_probe_bytes=oi_network['network_downloaded_bytes'],
                                             network_probe_evidence_sha256=oi_network['evidence_artifact_sha256'],
                                             full_pixel_coverage_audited=False,
                                             next='Full-ID retry/completeness and cumulative byte/output journal; protected pixel audit and method/prompt/decoder/evaluator locks before TEST inference')
pixel_index_path = root / 'reports/VG_PIXEL_FINGERPRINT_PROGRESS.json'
if pixel_index_path.exists():
    pixel_index = json.loads(pixel_index_path.read_text(encoding='utf8'))
    assert not pixel_index['model_inference_or_benchmark_scoring'] and not pixel_index['vg_images_extracted']
    state['vg_pixel_fingerprint_preparation'] = pixel_index
    if pixel_index['indexed_images'] < pixel_index['target_images']:
        state['next_independent_actions'][1] = 'Run one bounded remote_vg_pixel_fingerprint_batch.py per heartbeat in project environment, resume pinned SQLite journal; then compare protected pixel fingerprints before final clean release'
qwen_quant_path = root / 'reports/QWEN_QUANTIZATION_PREFLIGHT.json'
if qwen_quant_path.exists():
    qwen_quant = json.loads(qwen_quant_path.read_text(encoding='utf8'))
    assert qwen_quant['status'] == 'HEADER_ONLY_QUANTIZATION_PREFLIGHT_COMPLETE'
    assert not qwen_quant['model_weights_loaded'] and not qwen_quant['cuda_used']
    assert not qwen_quant['resource_gate_passed'] and not qwen_quant['benchmark_test_scored']
    state['qwen_quantization_preparation'] = qwen_quant
caption_stats_path = root / 'reports/CAPTION_STATISTICS_PROBE.json'
if caption_stats_path.exists():
    caption_stats = json.loads(caption_stats_path.read_text(encoding='utf8'))
    assert caption_stats['status'] == 'PASS' and not caption_stats['benchmark_test_scored']
    assert not caption_stats['references_or_benchmark_scores_read']
    state['caption_statistics_preparation'] = caption_stats
external_pixels_path = root / 'reports/EXTERNAL_PIXEL_FINGERPRINT_PROGRESS.json'
if external_pixels_path.exists():
    external_pixels = json.loads(external_pixels_path.read_text(encoding='utf8'))
    assert not external_pixels['benchmark_test_scored'] and not external_pixels['model_or_metric_run']
    assert external_pixels['network_downloaded_bytes'] == 0
    state['external_pixel_fingerprint_preparation'] = external_pixels
    state['next_independent_actions'][2] = 'Complete existing protected external pixel index, compare exact/perceptual candidates with VG; prepare bounded no-cache OI fingerprint streaming and authorized missing COCO/Flickr pixels'
exact_pixels_path = root / 'reports/CAPTION_EXACT_PIXEL_COMPARISON.json'
if exact_pixels_path.exists():
    exact_pixels = json.loads(exact_pixels_path.read_text(encoding='utf8'))
    assert exact_pixels['status'] == 'PARTIAL_EXACT_PIXEL_COMPARISON_COMPLETE'
    assert not exact_pixels['fingerprint_journals_modified'] and not exact_pixels['training_manifest_modified']
    assert not exact_pixels['benchmark_test_scored'] and not exact_pixels['clean_release_certified']
    state['caption_exact_pixel_comparison'] = exact_pixels
    state['caption_readiness']['pending_exact_pixel_exclusions_before_fresh_training'] = exact_pixels.get('retained_draft_ids_with_exact_protected_matches',{})
    state['caption_exact_pixel_comparison']['matches_latest_index_coverage'] = (
        exact_pixels['vg_indexed_images'] == state['vg_pixel_fingerprint_preparation']['indexed_images']
        and exact_pixels['external_indexed_images'] == state['external_pixel_fingerprint_preparation']['indexed_images'])
    state['next_independent_actions'][2] = 'Refresh read-only exact pixel comparison when journal coverage changes; verify perceptual candidates independently and complete protected external pixels including bounded no-cache OI streaming'
metric_runtime_path = root / 'reports/CAPTION_METRIC_RUNTIME_PROBE.json'
if metric_runtime_path.exists():
    metric_runtime = json.loads(metric_runtime_path.read_text(encoding='utf8'))
    assert metric_runtime['status'] == 'PASS' and not metric_runtime['benchmark_test_scored']
    assert not metric_runtime['shared_package_files_modified'] and not metric_runtime['references_or_benchmark_scores_read']
    state['caption_metric_runtime_preparation'] = metric_runtime
    state['caption_runtime_preparation'].update(
        repeat_safe_private_java8_adapter_verified=True,
        private_runtime_module_sha256=metric_runtime['runtime_source_sha256'],
        repeated_actual_spice_calls=metric_runtime['repeated_actual_spice_calls'],
        full_corpus_protocol_verified=False)
corpus_metrics_path = root / 'reports/CAPTION_CORPUS_METRIC_PROBE.json'
if corpus_metrics_path.exists():
    corpus_metrics = json.loads(corpus_metrics_path.read_text(encoding='utf8'))
    assert corpus_metrics['status'] == 'PASS' and not corpus_metrics['benchmark_test_scored']
    assert not corpus_metrics['references_or_benchmark_scores_read'] and not corpus_metrics['shared_package_files_modified']
    state['caption_corpus_metric_preparation'] = corpus_metrics
    state['caption_runtime_preparation'].update(
        synthetic_corpus_wrapper_native_oracle_verified=True,
        corpus_metric_wrapper_sha256=corpus_metrics['source_hashes']['src/rsgformer/caption_metrics.py'],
        spice_batch_vs_full_fixture_verified=True,
        full_corpus_protocol_verified=False)
native_gpu_path = root / 'reports/FRESH_DETECTOR_NATIVE_GPU_PROBE.json'
if native_gpu_path.exists():
    native_gpu = json.loads(native_gpu_path.read_text(encoding='utf8'))
    assert not native_gpu['dataset_training_launched'] and not native_gpu['checkpoint_saved']
    assert not native_gpu['clean_release_certified'] and not native_gpu['live_v2_modified']
    state['fresh_detector_native_gpu_probe'] = native_gpu
    state['fresh_detector_preparation'].update(
        native_resolution_gpu_smoke_status=native_gpu['status'],
        native_resolution_gpu_resource_gate_passed=native_gpu['status']=='PASS',
        actual_clean_dataset_training_launched=False,
        next='Preserve first FP32 native-resolution OOM; prepare activation-checkpoint gradient parity on CPU and admit any future GPU version using live reserved/process memory as well as allocated bytes')
checkpoint_cpu_path = root / 'reports/FRESH_CHECKPOINT_CPU_PROBE.json'
if checkpoint_cpu_path.exists():
    checkpoint_cpu = json.loads(checkpoint_cpu_path.read_text(encoding='utf8'))
    assert not checkpoint_cpu['cuda_used'] and not checkpoint_cpu['dataset_training_launched']
    assert not checkpoint_cpu['benchmark_test_scored'] and not checkpoint_cpu['live_v2_modified']
    assert not checkpoint_cpu['native_resolution_resource_gate_passed']
    state['fresh_detector_activation_checkpoint_preparation'] = checkpoint_cpu
    if checkpoint_cpu['status'] == 'PASS':
        state['fresh_detector_preparation'].update(
            activation_checkpoint_low_resolution_cpu_parity='PASS',
            native_resolution_gpu_resource_gate_passed=False,
            next='Preserve original native GPU OOM and checkpoint CPU evidence; admit separately versioned GPU parity/optimizer probe only after measuring aggregate reserved/process footprints; final clean pixel release and training preregistration still required')
oi_stream_path = root / 'reports/OI_STREAM_FINGERPRINT_PILOT.json'
if oi_stream_path.exists():
    oi_stream = json.loads(oi_stream_path.read_text(encoding='utf8'))
    assert not oi_stream['benchmark_test_scored'] and not oi_stream['model_or_metric_run']
    assert not oi_stream['reference_captions_read'] and not oi_stream['oi_train_or_val_accessed']
    assert not oi_stream['live_v2_modified'] and oi_stream['disk_pixel_bytes_written'] == 0
    assert oi_stream['pilot_target_images'] == 20 and oi_stream['indexed_images'] <= 20
    assert not oi_stream['full_protected_pixel_coverage'] and not oi_stream['clean_release_certified']
    state['openimages_stream_fingerprint_pilot'] = oi_stream
    state['openimages_test_readiness'].update(
        stream_fixed_pilot_status=oi_stream['status'],
        stream_fixed_pilot_indexed_images=oi_stream['indexed_images'],
        stream_fixed_pilot_persistent_network_charge_bytes=oi_stream['persistent_network_charge_bytes'],
        full_pixel_coverage_audited=False,
        next='Verify durable retry/crash/byte-ledger fixtures; separately register full TEST streaming budget/version; join protected snapshots before final clean release')
oi_ledger_path = root / 'reports/OI_STREAM_LEDGER_PROBE.json'
if oi_ledger_path.exists():
    oi_ledger = json.loads(oi_ledger_path.read_text(encoding='utf8'))
    assert oi_ledger['status'] == 'PASS' and oi_ledger['network_downloaded_bytes'] == 0
    assert not oi_ledger['real_pixel_journals_modified'] and not oi_ledger['benchmark_test_scored']
    assert not oi_ledger['live_v2_modified'] and not oi_ledger['full_streaming_registered']
    state['openimages_stream_ledger_preparation'] = oi_ledger
    state['openimages_test_readiness'].update(
        stream_byte_retry_crash_lock_fixture='PASS',
        full_pixel_coverage_audited=False,
        next='Register separately budgeted full TEST streaming version/continuation policy; join validated protected fingerprints with VG before final clean release')
oi_full_fixture_path = root / 'reports/OI_FULL_STREAM_LEDGER_PROBE.json'
if oi_full_fixture_path.exists():
    full_fixture = json.loads(oi_full_fixture_path.read_text(encoding='utf8'))
    assert full_fixture['status']=='PASS' and full_fixture['external_owned_worker_timeout']
    assert full_fixture['elapsed_seconds'] < 120 and full_fixture['network_downloaded_bytes']==0
    assert not full_fixture['real_pixel_journals_modified'] and not full_fixture['live_v2_modified']
    state['openimages_full_stream_ledger_preparation'] = full_fixture
oi_launch_path = root / 'reports/OI_STREAM_PHASE_LAUNCH.json'
if oi_launch_path.exists():
    full_launch = json.loads(oi_launch_path.read_text(encoding='utf8'))
    assert not full_launch['benchmark_test_scored'] and not full_launch['model_or_metric_run']
    assert not full_launch['live_v2_modified'] and not full_launch['clean_release_certified']
    state['openimages_full_stream_phase'] = {'launch':full_launch,
                                           'process_observation':snapshot.get('oi_stream_audit'),
                                           'process_observation_utc':snapshot['utc']}
    launch_check_path=root/'reports/OI_STREAM_LAUNCH_CHECK.json'
    if launch_check_path.exists():
        launch_check=json.loads(launch_check_path.read_text(encoding='utf8'))
        assert launch_check['preregistration_sha256']==full_launch['preregistration_sha256']
        state['openimages_full_stream_phase']['launch_verification']=launch_check
    state['openimages_test_readiness'].update(
        full_test_identity_streaming_registered=True,
        full_pixel_coverage_audited=False,
        next='Wait for bounded full TEST identity stream outcome; join protected VG/OI snapshots and independently verify perceptual candidates before final clean release')
    for filename,key in [('OI_STREAM_FINGERPRINT_PROGRESS.json','latest_progress'),
                         ('OI_STREAM_PHASE_RESULT.json','terminal_result')]:
        candidate = root / 'reports' / filename
        observed=snapshot.get(filename)
        if observed:
            current=json.loads(candidate.read_text(encoding='utf8')) if candidate.exists() else {}
            if observed.get('utc','') > current.get('utc',''):
                assert not observed['benchmark_test_scored'] and not observed['clean_release_certified']
                candidate.write_text(json.dumps(observed,indent=2),encoding='utf8')
        if candidate.exists():
            value=json.loads(candidate.read_text(encoding='utf8'))
            assert not value['benchmark_test_scored'] and not value['clean_release_certified']
            state['openimages_full_stream_phase'][key]=value
    state['next_independent_actions'][2]='Continue bounded full OI TEST identity stream; refresh complete protected exact/perceptual joins and preserve failed IDs before a versioned clean release'
exact_v2_path=root/'reports/CAPTION_EXACT_PIXEL_COMPARISON_V2.json'
if exact_v2_path.exists():
    exact_v2=json.loads(exact_v2_path.read_text(encoding='utf8'))
    assert exact_v2['status']=='PARTIAL_EXACT_PIXEL_COMPARISON_COMPLETE'
    assert not exact_v2['fingerprint_journals_modified'] and not exact_v2['training_manifest_modified']
    assert not exact_v2['benchmark_test_scored'] and not exact_v2['clean_release_certified']
    assert exact_v2['script_sha256']==hashlib.sha256((root/'scripts/remote_caption_exact_pixel_compare_v2.py').read_bytes()).hexdigest()
    state['caption_exact_pixel_comparison_with_oi_stream']=exact_v2
    state['caption_exact_pixel_comparison_with_oi_stream']['matches_latest_vg_index_coverage']=(
        exact_v2['vg_indexed_images']==state['vg_pixel_fingerprint_preparation']['indexed_images'])
    pending=state['caption_readiness'].get('pending_exact_pixel_exclusions_before_fresh_training',{})
    for split,identities in exact_v2['retained_draft_ids_with_exact_protected_matches'].items():
        pending[split]=sorted(set(pending.get(split,[])).union(identities))
    state['caption_readiness']['pending_exact_pixel_exclusions_before_fresh_training']=pending
    state['next_independent_actions'][2]='Refresh pinned remote_caption_exact_pixel_compare_v2.py after each VG batch, readonly union with full OI streaming journal; complete coverage and perceptual audits before final exclusions/fresh training'
for filename,key in [('OI_EXHAUSTED_SOURCE_AUDIT.json','openimages_retry_exhausted_source_evidence'),
                     ('OI_FAILED_HEADER_PROBE_V2.json','openimages_failed_source_header_preparation'),
                     ('OI_FAILED_HEADER_PROBE_V3.json','openimages_large_primary_header_preparation'),
                     ('OI_READER_V2_CPU_PROBE.json','openimages_reader_v2_compatibility_preparation'),
                     ('OI_MPO_REPAIR_PILOT.json','openimages_mpo_compatibility_pilot')]:
    candidate=root/'reports'/filename
    if candidate.exists():
        value=json.loads(candidate.read_text(encoding='utf8'))
        assert not value['benchmark_test_scored'] and not value['live_v2_modified']
        state[key]=value
exact_v3_path=root/'reports/CAPTION_EXACT_PIXEL_COMPARISON_V3.json'
if exact_v3_path.exists():
    exact_v3=json.loads(exact_v3_path.read_text(encoding='utf8'))
    assert exact_v3['status']=='PARTIAL_EXACT_PIXEL_COMPARISON_COMPLETE'
    assert exact_v3['script_sha256']==hashlib.sha256((root/'scripts/remote_caption_exact_pixel_compare_v3.py').read_bytes()).hexdigest()
    assert exact_v3['supplemental_evidence_sha256']=='82ee44248109203c7c363a9ef2b19cf1bbbc6dfdece284bc7eed5b7afe832d34'
    assert exact_v3['supplemental_repair_fingerprints_joined']
    assert not any(exact_v3[key] for key in ('fingerprint_journals_modified','training_manifest_modified','benchmark_test_scored','clean_release_certified'))
    state['caption_exact_pixel_comparison_with_supplemental_compatibility']=exact_v3
    exact_v3['matches_latest_vg_index_coverage']=(exact_v3['vg_indexed_images']==state['vg_pixel_fingerprint_preparation']['indexed_images'])
    pending=state['caption_readiness'].get('pending_exact_pixel_exclusions_before_fresh_training',{})
    for split,identities in exact_v3['retained_draft_ids_with_exact_protected_matches'].items():
        pending[split]=sorted(set(pending.get(split,[])).union(identities))
    state['caption_readiness']['pending_exact_pixel_exclusions_before_fresh_training']=pending
    state['next_independent_actions'][2]='Refresh pinned remote_caption_exact_pixel_compare_v3.py after each VG batch; readonly local/stream/supplemental identity union; preserve all failed IDs; final coverage and perceptual gates before fresh training'
if state.get('openimages_retry_exhausted_source_evidence',{}).get('retry_exhausted_images',0)>0:
    joined=state.get('caption_exact_pixel_comparison_with_supplemental_compatibility',{})
    repaired=set(joined.get('supplemental_compatibility_ids',[]))
    exhausted={row['image_id'] for row in state['openimages_retry_exhausted_source_evidence'].get('records',[])}
    state['openimages_test_readiness'].update(
        original_stream_has_retry_exhausted_sources=True,
        failed_ids_preserved_in_denominator=True,
        supplemental_repair_fingerprints_joined=joined.get('supplemental_repair_fingerprints_joined',False),
        supplemental_repair_ids=sorted(repaired),
        retry_exhausted_ids_still_missing_supplemental_fingerprints=sorted(exhausted-repaired),
        all_retry_exhausted_sources_repaired=False,
        next='Preserve v1 failures/charges; version JPEG/MPO primary-frame compatibility and supplemental fingerprint union for every failed ID before complete-set inference; no missing-ID omission')
large_headers=state.get('openimages_large_primary_header_preparation',{})
if large_headers.get('status')=='HEADER_AUDIT_COMPLETE':
    blocked=[{'image_id':row['image_id'],'dimensions':row['header_dimensions'],
              'primary_pixels':row['header_dimensions'][0]*row['header_dimensions'][1]}
             for row in large_headers['records'] if row['header_dimensions'][0]*row['header_dimensions'][1]>16000000]
    state['openimages_test_readiness']['known_primary_pixel_cap_blocked_sources']=blocked
    state['openimages_test_readiness']['primary_frame_pixel_cap']=16000000
    state['openimages_test_readiness']['large_sources_full_body_retry_launched']=False
geometry_probe_path=root/'reports/VG_REGION_CAPTION_GEOMETRY_PROBE.json'
if geometry_probe_path.exists():
    geometry_probe=json.loads(geometry_probe_path.read_text(encoding='utf8'))
    assert geometry_probe['status']=='PASS' and not geometry_probe['caption_evaluation_admitted']
    assert geometry_probe['source_sha256']==hashlib.sha256((root/'src/rsgformer/vg_caption_region_units.py').read_bytes()).hexdigest()
    assert geometry_probe['protocol_sha256']==hashlib.sha256((root/'configs/vg_region_caption_geometry_v1.json').read_bytes()).hexdigest()
    state['vg_region_caption_geometry_preparation']=geometry_probe
    state['caption_readiness']['provided_crop_geometry_rule_locked']=True
    state['caption_readiness']['learned_region_caption_matching_protocol_locked']=False
    state['next_independent_actions'][0]='Provided-crop geometry v1 locked and synthetic PASS; complete bounded source schema/eligibility audit using audited raw dimensions, then lock actual unit membership/workload and independent dev prompts before evaluation'
geometry_schema_path=root/'reports/VG_REGION_CAPTION_SCHEMA_PROBE.json'
if geometry_schema_path.exists():
    geometry_schema=json.loads(geometry_schema_path.read_text(encoding='utf8'))
    assert geometry_schema['status']=='PARTIAL_REFERENCE_SCHEMA_GEOMETRY_PROBE_PASS'
    assert not any(geometry_schema[key] for key in ('caption_evaluation_admitted','benchmark_test_scored','live_v2_modified'))
    assert geometry_schema['source_hashes']['adapter']==hashlib.sha256((root/'src/rsgformer/vg_caption_region_units.py').read_bytes()).hexdigest()
    assert geometry_schema['source_hashes']['geometry_protocol']==hashlib.sha256((root/'configs/vg_region_caption_geometry_v1.json').read_bytes()).hexdigest()
    state['vg_region_caption_partial_schema_preparation']=geometry_schema
speed_launch=root/'reports/VG_SPEEDUP_LAUNCH.json'
if speed_launch.exists():
    launch=json.loads(speed_launch.read_text(encoding='utf8'))
    fixture=json.loads((root/'reports/VG_SPEEDUP_FIXTURE.json').read_text(encoding='utf8'))
    assert fixture['status']=='PASS'
    assert fixture['module_sha256']==launch['protocol']['source_hashes']['src/rsgformer/vg_speedup.py']
    assert not launch['scientific_change_applied'] and not launch['benchmark_test_accessed']
    speed_status=root/'reports/VG_SPEEDUP_LAST_STATUS.json'
    state['training_speedup_preparation']={
        'launch_status':launch['status'],'controller_pid':launch['controller_pid'],
        'preregistration_sha256':launch['preregistration_sha256'],
        'original_wall_deadline_unix':launch['protocol']['original_wall_deadline_unix'],
        'fixture_status':fixture['status'],'fixture_scope':'CPU synthetic slicing/loss/gradient contracts, not full native GPU parity',
        'last_status':json.loads(speed_status.read_text(encoding='utf8')) if speed_status.exists() else None,
        'scientific_change_applied':False,'benchmark_test_accessed':False,
        'next':'Read once-only worker result on hourly heartbeat; new matched four-arm protocol required before microbatch adoption; keep absolute original wall cap and preserve old evidence'}
    adoption_path=root/'reports/VG_SPEEDUP_ADOPTION.json'
    if adoption_path.exists():
        adoption=json.loads(adoption_path.read_text(encoding='utf8'))
        state['training_speedup_preparation'].update(adoption=adoption,
            scientific_change_applied=adoption.get('scientific_change_applied',False),
            next='Monitor active optimized version once hourly; preserve original native as historical only; no duplicate old worker/benchmark or blind partial resume')
if state['vg_pixel_fingerprint_preparation']['status']=='VG_PIXEL_INDEX_COMPLETE':
    state['next_independent_actions'][1]='VG index complete: skip batch; refresh protected exact comparison when external coverage changes, then complete independently locked perceptual audit before final fresh release'
perceptual_policy_path=root/'reports/CAPTION_PERCEPTUAL_POLICY_FIXTURE_V1.json'
if perceptual_policy_path.exists():
    policy_fixture=json.loads(perceptual_policy_path.read_text(encoding='utf8'))
    assert policy_fixture['status']=='PASS' and not policy_fixture['clean_release_certified']
    for name,expected in policy_fixture['source_hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    state['caption_perceptual_policy_preparation']=policy_fixture
    state['next_independent_actions'][1]='VG index complete; prepare separately bounded read-only source snapshots and private candidate index using locked caption_perceptual_exclusion_v1; no existing journal edits, final manifest or clean certification before all gates'
internal_fixture_path=root/'reports/VG_INTERNAL_PHASH_BATCH_FIXTURE_V1.json'
if internal_fixture_path.exists():
    internal_fixture=json.loads(internal_fixture_path.read_text(encoding='utf8'))
    assert internal_fixture['status']=='PASS' and not internal_fixture['real_fingerprints_accessed']
    for name,expected in internal_fixture['source_hashes'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
    state['vg_internal_perceptual_batch_fixture']=internal_fixture
    internal_progress_path=root/'reports/VG_INTERNAL_PHASH_PROGRESS.json'
    if internal_progress_path.exists():
        internal_progress=json.loads(internal_progress_path.read_text(encoding='utf8'))
        assert not any(internal_progress[key] for key in ('clean_release_certified','training_manifest_modified','test_scored'))
        if internal_progress['status']!='VG_INTERNAL_PHASH_AUDIT_BLOCKED':
            for name,expected in internal_progress['source_hashes'].items():
                assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected,name
            assert internal_progress['journal_source_unchanged']
        state['vg_internal_perceptual_candidate_preparation']=internal_progress
        if internal_progress['status']=='VG_INTERNAL_PHASH_AUDIT_IN_PROGRESS':
            state['next_independent_actions'][1]='One pinned remote_vg_internal_phash_batch_v1.py batch per permitted12-hour heartbeat; private journal resume and original VG journal read-only; complete internal queries and external/perceptual coverage before finalized exclusions/fresh training'
        elif internal_progress['status']=='VG_INTERNAL_PHASH_CANDIDATES_COMPLETE':
            state['next_independent_actions'][1]='Fetch and SHA-verify immutable completed internal candidate evidence; retain conservative exclusions for a new release, then finish protected external coverage/perceptual audit; internal completion alone is not clean certification'
        else:
            state['next_independent_actions'][1]='Audit bounded internal candidate phase error with preserved private journal/preregistration; do not edit locked phase or resume changed source; continue independent external preparation'
comparison_script=root/'scripts/update_vg_speed_validation_comparison.py'
if active_experiment=='vg_method_v2_speed_m2_v1' and comparison_script.exists():
    import runpy
    runpy.run_path(str(comparison_script),run_name='__main__')
    sgdet_state['screening_validation_comparison']=json.loads((root/'reports/VG_V2_SPEED_VALIDATION_COMPARISON.json').read_text(encoding='utf8'))
# Preserve completion conclusions and current cadence in the independent program ledger.
state['next_independent_actions']=[action.replace('per hourly heartbeat','per permitted12-hour heartbeat').replace('after each VG batch','only after protected external coverage changes') for action in state['next_independent_actions']]
outcome_path=root/'reports/VG_V2_SPEED_SCREENING_OUTCOME.json'
if outcome_path.exists() and active_experiment=='vg_method_v2_speed_m2_v1':
    outcome=json.loads(outcome_path.read_text(encoding='utf8'))
    assert outcome['experiment_id']==active_experiment and not outcome['test_scored']
    sgdet_state['screening_stage_conclusion']=outcome
    if snapshot.get('active_experiment',{}).get('execution_revision'):
        sgdet_state['screening_stage_conclusion_is_preserved_pre_extension_outcome']=True
component_status=root/'reports/VG_COMPONENT_CONTROL_TEST_LAST_STATUS.json'
if component_status.exists():
    component=json.loads(component_status.read_text(encoding='utf8'))
    phase_lock=root/'reports/VG_COMPONENT_CONTROL_TEST_PREREGISTRATION.json'
    phase=json.loads(phase_lock.read_text(encoding='utf8'))
    assert hashlib.sha256(phase_lock.read_text(encoding='utf8').encode()).hexdigest()==component['preregistration_sha256']
    for name,expected in phase['source_hashes'].items():
        candidate=root/name
        if candidate.exists():assert hashlib.sha256(candidate.read_bytes()).hexdigest()==expected,name
    state['component_control_test']={'phase_id':component['phase_id'],'snapshot':component,
                                   'preregistration_sha256':component['preregistration_sha256'],
                                   'selected_checkpoint':phase['checkpoint'],
                                   'authorization':phase['authorization'],'local_locked_hashes_verified':True}
    sgdet_state['historical_after_human_component_test_supersession']=True
    state['program_status']='COMPONENT_CONTROL_TEST_PHASE_CAPTION_PREPARATION_PENDING'
    state['next_independent_actions'][0]='Monitor the single Context component phase every12h; report one completed test/meaningful blocker; never resume the superseded confirmation or reuse test for tuning'
    result_path=root/'reports/VG_COMPONENT_CONTROL_TEST_RESULT.json'
    verification_path=root/'reports/VG_COMPONENT_CONTROL_TEST_COMPLETION_VERIFICATION.json'
    if result_path.exists() and verification_path.exists():
        completed=json.loads(result_path.read_text(encoding='utf8'))
        verified=json.loads(verification_path.read_text(encoding='utf8'))
        assert completed['status']=='COMPONENT_TEST_COMPLETE' and verified['status']=='PASS'
        assert hashlib.sha256(result_path.read_text(encoding='utf8').encode()).hexdigest()==verified['result_sha256']
        assert completed['checkpoint_sha256']==phase['checkpoint_sha256']==verified['checkpoint_sha256']
        state['component_control_test'].update(result=completed,completion_verification=verified,
                                              status='COMPLETE_SINGLE_TEST_DO_NOT_RERUN',test_scored=True)
        state['program_status']='COMPONENT_CONTROL_TEST_COMPLETE_CAPTION_PREPARATION_PENDING'
        state['next_independent_actions'][0]='Context single TEST complete and verified; preserve result and published comparison, no routine terminal-job polling, test rerun/tuning or blind confirmation resume; continue independent caption-clean preparation'
caption_check=root/'reports/VG_CAPTION_PREPARATION_LAST_CHECK.json'
if caption_check.exists():
    checked=json.loads(caption_check.read_text(encoding='utf8'))
    state['caption_preparation_last_check']=checked
    if checked.get('status')=='BLOCKED_DISK_RESERVE':
        state['program_status']='COMPONENT_TEST_COMPLETE_CAPTION_PREPARATION_BLOCKED_STORAGE'
        state['next_independent_actions'][0]='Restore at least16GiB disk reserve plus128MiB private journal admission before further caption preparation; do not delete preserved assets/checkpoints or lower cap; automation paused pending resource restoration'
cleanup_path=root/'reports/VG_V1_CACHE_CLEANUP_RESULT.json'
if cleanup_path.exists():
    cleanup=json.loads(cleanup_path.read_text(encoding='utf8'))
    if cleanup.get('status')=='COMPLETE':
        assert cleanup['deleted_file_count']==480 and cleanup['cache_directory_kept']
        assert cleanup['manifest_preserved_sha256']=='d39354aa61511c2bc3a02fa0b0115a1ad6473c2a48fb8ffcc4936043de95519e'
        state['authorized_v1_feature_cache_retirement']=cleanup
        state['sgdet_v1_cache_status']='RETIRED_WITH_HUMAN_AUTHORIZATION_NO_RESTART_KEEP_MANIFEST_AND_OUTCOME'
        if caption_check.exists() and checked.get('status')=='RESUMED_DISK_ADMISSION_RESTORED':
            state['program_status']='COMPONENT_CONTROL_TEST_COMPLETE_CAPTION_PREPARATION_PENDING'
            state['next_independent_actions'][0]='Continue bounded independent caption-clean audit and preparation after authorized V1 cache retirement; terminal Context TEST remains closed and original negative V1 must not restart'
color_launch=root/'reports/VG_COLOR_AUTO_FULL_LAUNCH_V1_20261025.json'
if color_launch.exists():
    launched=json.loads(color_launch.read_text(encoding='utf8'))
    checked_path=root/'reports/VG_COLOR_AUTO_FULL_LAST_STATUS_20261025.json'
    color_checked=json.loads(checked_path.read_text(encoding='utf8')) if checked_path.exists() else {}
    active=color_checked.get('status.json',{})
    state['color_automated_full_test']={
        'phase':'vg_color_auto_full_pipeline_v1_20261025','controller_pid':launched['controller_pid'],
        'preregistration_sha256':launched['preregistration_sha256'],
        'authorization':'25Oct user explicitly chose automated evaluation with limitations, then full colorTEST',
        'status':active.get('status','LAUNCHED'),'stage':active.get('stage','VAL_PARITY'),
        'last_snapshot':color_checked,'human_precision_gate_passed':False,'human_labels_created':False,
        'old_human_review_requirement_superseded_for_execution_only':True,
        'outputs_remain_unverified_color_candidates':True,'new_SGDet_test_inference_or_scoring':False,
        'TEST_GT_given_to_extractor':False,'caption_generation_started_by_this_phase':False,
        'automatic_retry_allowed':False,'full_test_scope':'all26446images,20eligibleclasses,max4requestednodes/image',
        'automatic_VAL_report':'reports/VG_COLOR_AUTO_VAL_DIAGNOSTIC_V1_20261025.json',
        'frozen_config':'configs/vg_color_automated_full_test_v1_20261025.json',
        'monitor':'scripts/remote_vg_color_auto_full_status_v1_20261025.py'}
    state['program_status']='AUTOMATED_COLOR_DIAGNOSTIC_FULL_TEST_PIPELINE_'+active.get('status','LAUNCHED')
    state['next_independent_actions'][0]='Let the immutable automated color controller finish VAL engineering parity, then the authorized one-pass full colorTEST cache and frozen weak-attribute diagnostic; no source edits, duplicate launch, threshold tuning, human-precision claim or terminal SGDet rerun'
    state['whole_program_complete']=False
    color_verified=root/'reports/VG_COLOR_AUTO_FULL_COMPLETION_VERIFICATION_20261026.json'
    color_conclusion=root/'reports/VG_COLOR_FULL_TEST_CONCLUSION_20261026.json'
    if color_verified.exists() and color_conclusion.exists():
        verified=json.loads(color_verified.read_text(encoding='utf8'))
        conclusion=json.loads(color_conclusion.read_text(encoding='utf8'))
        assert verified['status']=='PASS_LOCAL_COMPLETION_SOURCE_LEDGER_CACHE_AND_FULL_SG_STRUCTURE'
        assert conclusion['images']==26446 and conclusion['candidate_nodes']==5214
        assert conclusion['verification']['archive_sha256']==verified['archive_sha256']
        state['color_automated_full_test'].update(status='COMPLETE_VERIFIED_CANDIDATE_ONLY_DO_NOT_RERUN',
            current_active_compute=None,completion_verification=verified,conclusion=conclusion,
            terminal_phase_polling_allowed=False,
            local_cache='artifacts/vg_color_full_test_auto_v1_20261025/full_sg_color_candidates.jsonl.gz')
        state['program_status']='COLOR_TEST_COMPLETE_VERIFIED_CAPTION_VAL_PREPARATION_PENDING'
        state['next_independent_actions'][0]='Follow research/VG_CAPTION_AFTER_COLOR_TEST_PLAN_20261026.md: prepare a new preregistered TRAIN/VAL six-arm caption phase using frozen V3 candidate-color adapters, never launch old V2 candidate scripts, do not poll terminal color/SGDet or reopen TEST; caption TEST needs its own gate'
pilot_launch=root/'reports/VG_CAPTION_PILOT100_LAUNCH_20261026.json'
if pilot_launch.exists():
    launched=json.loads(pilot_launch.read_text(encoding='utf8'))
    current_path=root/'reports/VG_CAPTION_PILOT100_LAST_STATUS_20261026.json'
    current=json.loads(current_path.read_text(encoding='utf8')) if current_path.exists() else {}
    active=current.get('status.json',{})
    state['caption_pilot100']={
        'phase':'vg_caption_pilot100_v1_20261026','authorization':'26Oct human asked for captions and preliminary evaluation on100images across configurations',
        'controller_pid':launched['controller_pid'],'preregistration_sha256':launched['preregistration_sha256'],
        'status':active.get('status','LAUNCHED'),'stage':active.get('stage','prepare'),'last_snapshot':current,
        'image_count':100,'provided_regions_per_image':1,'configuration_count':12,'maximum_outputs':1200,
        'split':'VAL_pilot_not_TEST','candidate_color_only':True,'human_color_precision_gate_passed':False,
        'automatic_TEST_or_prompt_selection':False,'paused_seeds_resumed':False,
        'monitor':'scripts/remote_vg_caption_pilot100_status_v1_20261026.py',
        'plan':'research/VG_CAPTION_PILOT100_PREREGISTRATION_20261026.md'}
    state['program_status']='CAPTION_PILOT100_'+active.get('status','LAUNCHED')
    state['next_independent_actions'][0]='Let the immutable pilot100 controller cache fixed TRAIN/VAL candidate colors, generate all12configurations and score all100provided regions per corpus. Fetch and SHA-verify results after completion; no duplicate launch, oldV2 scripts, TEST opening or paused-seed resume'
    state['whole_program_complete']=False
    pilot_verified=root/'reports/VG_CAPTION_PILOT100_COMPLETION_VERIFICATION_20261026.json'
    pilot_result=root/'reports/VG_CAPTION_PILOT100_RESULT_20261026.json'
    if pilot_verified.exists() and pilot_result.exists():
        verification=json.loads(pilot_verified.read_text(encoding='utf8'))
        result=json.loads(pilot_result.read_text(encoding='utf8'))
        assert verification['status']=='PASS_PILOT100_ARCHIVE_SOURCE_INPUT_GENERATION_AND12_CORPUS_COMPLETION'
        assert verification['images']==100 and verification['unique_outputs']==1200 and len(result['scores_display_x100'])==12
        state['caption_pilot100'].update(status='COMPLETE_VERIFIED_ALL12_CONFIGS_DO_NOT_RERUN',stage='COMPLETE',current_active_compute=None,
            terminal_phase_polling_allowed=False,completion_verification=verification,result=result,
            conclusion='research/VG_CAPTION_PILOT100_CONCLUSION_20261026.md',local_output_directory='outputs/vg_caption_pilot100_20261026')
        state['program_status']='CAPTION_PILOT100_COMPLETE_VALIDATION_DIAGNOSIS_PENDING'
        state['next_independent_actions'][0]='Audit region-to-SG focus/missing objects/duplicate instances and sparse color coverage on TRAIN/dev; current pilot is observed evidence, no automatic winner selection or caption TEST. Any scientific repair requires a new finite version/prereg and independent check; preserve all completed outputs'
audit_report=root/'reports/VG_REGION_FOCUS_AUDIT_V1_20261027.json'
audit_verified=root/'reports/VG_REGION_FOCUS_AUDIT_VERIFICATION_20261027.json'
if audit_report.exists() and audit_verified.exists():
    audit=json.loads(audit_report.read_text(encoding='utf8'))
    checked=json.loads(audit_verified.read_text(encoding='utf8'))
    assert checked['status']=='PASS_LOCAL_RAW_AUDIT_SOURCE_AND900UNIT_EVIDENCE'
    assert checked['archive_sha256']==audit['archive_sha256']
    state['region_focus_audit']={'status':'COMPLETE_VERIFIED_NO_POLICY_CHANGE','result':audit,'verification':checked,'proposal':'research/VG_REGION_FOCUS_REPAIR_PROPOSAL_20261027.md','current_active_compute':None}
    state['program_status']='REGION_FOCUS_AUDIT_COMPLETE_NEW_VERSION_PREPARATION_PENDING'
    state['next_independent_actions'][0]='Preregister the finite TRAIN-only L0/L1/L2 geometry-focus comparison in a new source namespace, then untouched caption_check once; fewer empty regions is not semantic correctness. No edits to locked historical code, no TEST tuning or automatic TEST launch'
focus_launch=root/'reports/VG_FOCUS_REPAIR_LAUNCH_20261028.json'
if focus_launch.exists():
    focus=json.loads(focus_launch.read_text(encoding='utf8'))
    if focus.get('status')=='LAUNCHED':
        focus_status_path=root/'reports/VG_FOCUS_REPAIR_LAST_STATUS_20261028.json'
        focus_checked=json.loads(focus_status_path.read_text(encoding='utf8')) if focus_status_path.exists() else {}
        focus_active=focus_checked.get('status.json',{})
        state['focus_repair_experiment']={'status':focus_active.get('status','LAUNCHED'),'controller_pid':focus['controller_pid'],'preregistration_sha256':focus['preregistration_sha256'],'last_snapshot':focus_checked,'TEST_accessed':False,'policy_adopted':False}
        state['program_status']='FOCUS_REPAIR_'+focus_active.get('status','LAUNCHED')
        state['next_independent_actions'][0]='Allow the locked finite TRAIN-selection and frozen caption_check controller to complete; no duplicate launch, source edits, check reselection, TEST opening or seed resume'
    elif focus.get('status')=='BLOCKED_PRELAUNCH_GPU_ADMISSION':
        state['focus_repair_experiment']={'status':'PREPARED_CONTRACTS_PASS_WAITING_GPU','resource_block':focus,'current_active_compute':None,'TEST_accessed':False,'policy_adopted':False,'config':'configs/vg_focus_repair_v1_20261028.json'}
        state['program_status']='FOCUS_REPAIR_PREPARED_WAITING_GPU_HEADROOM'
        state['next_independent_actions'][0]='At the next due12h check measure resource admission once; if all pass, launch the unchanged exclusive focus-repair launcher once. Main remote namespace/report were verified absent; no Qwen process was started. Do not lower caps or kill unrelated jobs; dedup the existing GPU headroom block'
focus_failure_verified=root/'reports/VG_FOCUS_REPAIR_FAILURE_VERIFICATION_20261030.json'
if focus_failure_verified.exists():
    verified=json.loads(focus_failure_verified.read_text(encoding='utf8'))
    assert verified['status']=='PASS_RAW_FAILED_PARENT_SOURCES_TRAIN_GENERATION_METRICS_SELECTION' and not verified['check_access_opened']
    state['focus_repair_experiment'].update(status='FAILED_PARENT_PRESERVED_CHECK_UNOPENED_DO_NOT_RESTART',current_active_compute=None,terminal_phase_polling_allowed=False,failure_verification=verified)
    state['program_status']='FOCUS_REPAIR_TRAIN_LOCKED_CHECK_RESOURCE_BLOCKED'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Read resources once at next due12h; no terminal parent polling or server writes below admission. R1 locally prepared preserves all TRAIN evidence and original L2 selection; after all admission guards pass sync only new R1 files, then exclusive audited check continuation once'
    r1contract=root/'reports/VG_FOCUS_REPAIR_R1_LOCAL_CONTRACT_20261030.json'
    if r1contract.exists():
        state['focus_repair_R1']={'status':'PREPARED_LOCAL_ONLY_WAITING_RESOURCE_ADMISSION','contract':json.loads(r1contract.read_text(encoding='utf8')),'current_active_compute':None,'TEST_accessed':False,'check_opened':False,'policy_adopted':False,'preregistration':'research/VG_FOCUS_REPAIR_FAILURE_AND_R1_PREREGISTRATION_20261030.md'}
    r1launch=root/'reports/VG_FOCUS_REPAIR_R1_LAUNCH_20261030.json'
    if r1launch.exists():
        launch=json.loads(r1launch.read_text(encoding='utf8'))
        if launch.get('status')=='LAUNCHED':
            sp=root/'reports/VG_FOCUS_REPAIR_R1_LAST_STATUS_20261030.json'
            checked=json.loads(sp.read_text(encoding='utf8')) if sp.exists() else {}
            active=checked.get('status.json',{})
            state['focus_repair_R1'].update(status=active.get('status','LAUNCHED'),launch=launch,last_snapshot=checked)
            state['program_status']='FOCUS_REPAIR_R1_'+active.get('status','LAUNCHED')
            state['current_active_compute']='vg_focus_repair_r1_20261030' if active.get('status','LAUNCHED') in ['RUNNING','LAUNCHED'] else None
            state['focus_repair_R1']['current_active_compute']=state['current_active_compute']
            state['next_independent_actions'][0]='Let the immutable R1 check-only controller finish once; no parent TRAIN recomputation, duplicate launch, source edits, check reselection, TEST opening or seed resume'

r1_verified_path=root/'reports/VG_FOCUS_REPAIR_R1_COMPLETION_VERIFICATION_20261101.json'
r1_conclusion_path=root/'reports/VG_FOCUS_REPAIR_R1_CONCLUSION_20261101.json'
if r1_verified_path.exists() and r1_conclusion_path.exists():
    verified=json.loads(r1_verified_path.read_text(encoding='utf8'));conclusion=json.loads(r1_conclusion_path.read_text(encoding='utf8'))
    assert verified['status']=='PASS_R1_ARCHIVE_SOURCE_PARENT_REUSE_SELECTION_AND512_CHECK_OUTPUTS' and verified['check_outputs']==512 and verified['check_regions_per_policy']==256
    state['focus_repair_R1'].update(status='COMPLETE_VERIFIED_FROZEN_CHECK_NO_POLICY_ADOPTION',stage='COMPLETE',completion_verification=verified,conclusion=conclusion,terminal_phase_polling_allowed=False,current_active_compute=None,TEST_accessed=False,policy_adopted=False)
    state['program_status']='FOCUS_REPAIR_R1_COMPLETE_VERIFIED_FACTUALITY_REVIEW_PENDING'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Review fixed20 graph-to-text factuality packet and missing-target limitations locally; no terminal polling, automatic policy adoption, observed-check tuning, caption TEST or seed resume. New scientific work needs finite TRAIN/dev prereg and genuinely unused check'

offload_result_path=root/'reports/VG_V2_CACHE_OFFLOAD_RESULT_20261101.json'
if offload_result_path.exists():
    offload=json.loads(offload_result_path.read_text(encoding='utf8'))
    assert offload['status']=='COMPLETE_981_VERIFIED_V2_SHARDS_OFFLOADED_AND_SERVER_COPIES_REMOVED' and offload['deleted_files']==981
    state['V2_dense_feature_cache_storage']={'status':'OFFLOADED_LOCALLY_ORIGINAL_FP32_ARCHIVED_SERVER_SHARDS_REMOVED','result':offload,'server_manifest_preserved':True,'archive_directory':'artifacts/offloaded_vg_v2_features_20261101','backup_verification':'reports/VG_V2_CACHE_OFFLOAD_BACKUP_VERIFICATION_20261101.json','current_caption_R1_requires_dense_cache':False,'additional_SGG_seed_resume_requires_exact_cache_restore_or_separate_audited_adapter':True,'cache_restoration_requires_disk_reserve_admission':True,'paused_seeds_still_paused':True}

continuation_path=root/'reports/VG_V2_OFFLOAD_CONTINUATION_STATUS_20261101.json'
if continuation_path.exists():
    continuation=json.loads(continuation_path.read_text(encoding='utf8'))
    state['storage_offload_continuation']=continuation
    if continuation.get('status')!='RUNNING':state['current_active_maintenance']=None
    if continuation.get('status')=='RUNNING' and not (root/'reports/VG_V2_CACHE_OFFLOAD_RESULT_20261101.json').exists():
        state['program_status']='V2_CACHE_LOCAL_BACKUP_AND_GUARDED_CONTINUATION_ACTIVE'
        state['current_active_maintenance']='vg_v2_offload_continuation_20261101'
        state['next_independent_actions'][0]='Let the already-authorized owned local backup/continuation finish981SHA checks then exact server shard retirement and resource-gated exclusive R1; no duplicate transfer/deletion/launch and no terminal polling'

text_audit_path=root/'reports/VG_CAPTION_FIXED20_TEXT_SUPPORT_AUDIT_20261101.json'
if text_audit_path.exists():
    text_audit=json.loads(text_audit_path.read_text(encoding='utf8'))
    assert text_audit['status']=='FIXED20_ASSISTANT_TEXT_SUPPORT_AUDIT_COMPLETE_NOT_HUMAN_OR_VISUAL_ACCURACY' and text_audit['sample_units']==20 and text_audit['paired_outputs_reviewed']==40
    state['caption_fixed20_text_audit']={'status':text_audit['status'],'report':'reports/VG_CAPTION_FIXED20_TEXT_SUPPORT_AUDIT_20261101.json','conclusion':'research/VG_CAPTION_FIXED20_TEXT_SUPPORT_AUDIT_CONCLUSION_20261101.md','summary':text_audit['summary'],'human_labels_created':0,'source_images_viewed':0,'visual_truth_verified':False,'scientific_policy_changed_or_adopted':False,'current_active_compute':None}
    state['program_status']='FIXED20_TEXT_AUDIT_COMPLETE_TRAIN_DEV_GROUNDING_PREPARATION_PENDING'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Prepare a finite preregistered TRAIN/dev grounding diagnosis separating missing region target/facts from graph-to-text relation/state/material support; references evaluation-only, no observed-check tuning/automatic policy adoption/TEST launch. Reserve genuinely unused heldout membership and preserve exposure ledger'

grounding_verification_path=root/'reports/VG_GROUNDING_DIAGNOSTIC_VERIFICATION_20261101.json'
if grounding_verification_path.exists():
    verified=json.loads(grounding_verification_path.read_text(encoding='utf8'))
    assert verified['status']=='PASS_TRAIN2400_DIAGNOSTIC_HASHES_DENOMINATORS_AND_FRESH128_METADATA' and verified['saved_caption_rows']==2400
    state['grounding_TRAIN_diagnostic']={'status':'COMPLETE_VERIFIED_TEXT_PROXY_NOT_VISUAL_ACCURACY','verification':verified,'result':'reports/VG_GROUNDING_DIAGNOSTIC_RESULT_20261101.json','conclusion':'research/VG_GROUNDING_DIAGNOSTIC_CONCLUSION_20261101.md','new_scientific_policy_adopted':False,'current_active_compute':None}
    state['grounding_fresh_VAL_reservation']={'status':'PROVISIONAL_REFERENCE_BLIND_IDS_ONLY','config':'configs/vg_grounding_fresh_val_reservation_v2_20261101.json','dev_images':64,'check_images':64,'prior_VAL_parents_excluded':1528,'region_cache_prepared':False,'new_check_opened':False,'perceptual_audit_complete':False,'caption_clean_claimed':False}
    state['program_status']='GROUNDING_TRAIN_DIAGNOSIS_COMPLETE_FRESH_DEV_PREPARATION_PENDING'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Implement/audit/preregister finite G0 free-form versus G1 literal-renderer versus G2 existing-ID selection grounding comparison; use frozen TRAIN and fresh dev, references scoring-only. Prepare geometry-only dev cache from saved VAL graphs under hashes; reserve new check unopened until selection lock. No observed-check tuning, automatic adoption, TEST or seed resume. Draft research/VG_GROUNDING_NEXT_COMPARISON_DRAFT_20261101.md'

grounding_core_contract_path=root/'reports/VG_GROUNDING_CONTROLS_LOCAL_CONTRACT_20261102.json'
grounding_dev_verified_path=root/'reports/VG_GROUNDING_DEV_PREPARATION_VERIFICATION_20261102.json'
if grounding_core_contract_path.exists() and grounding_dev_verified_path.exists():
    contract=json.loads(grounding_core_contract_path.read_text(encoding='utf8'));verified=json.loads(grounding_dev_verified_path.read_text(encoding='utf8'))
    assert contract['status']=='PASS_NEW_GROUNDING_CORE_AND_CPU_PREPARATION_CONTRACTS_NO_GPU_LAUNCH' and contract['saved_TRAIN_graphs_and_G0_bindings_checked']==800
    assert verified['status']=='PASS_NEW_CORE_CONTRACTS_AND_FRESH64_DEV256_CACHE_SOURCE_INPUT_GEOMETRY_REFERENCE_SEPARATION' and verified['dev_regions']==256 and verified['check_parents_not_exported']==64
    state['grounding_controls_preparation']={'status':'CORE_FROZEN_CONTRACTS_AND_FRESH_DEV_CACHE_VERIFIED_MAIN_GPU_NOT_IMPLEMENTED','source_manifest':'configs/vg_grounding_controls_source_manifest_v1_20261102.json','source_manifest_sha256':verified['source_manifest_sha256'],'local_contract':contract,'dev_cache_verification':verified,'status_document':'research/VG_GROUNDING_CONTROLS_PREPARATION_STATUS_20261102.md','main_GPU_experiment_fully_implemented':False,'current_active_compute':None,'terminal_preparation_polling_allowed':False,'new_metrics_or_scientific_selection':False}
    state['grounding_fresh_VAL_reservation'].update(status='DEV64_CACHE_READY_CHECK64_REFERENCE_UNEXPORTED_NO_GENERATION_GATE',dev_regions=256,region_cache_prepared=True,check_cache_prepared=False,new_check_opened=False)
    state['program_status']='GROUNDING_CONTROLS_CORE_AND_DEV_CACHE_PREPARED_MAIN_IMPLEMENTATION_PENDING'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Implement/audit/preregister new G0/G1/G2 generation/scoring/controller/exclusive launcher source version before inference. Reuse frozen core and verified dev64/256cache; do not rerun terminal CPU prep. Maximum1312new Qwen calls (TRAIN G2 800,dev G0/G2 512), same bounded SG/native budgets, invalid selection empty retained. Source/tokenizer/runtime/resource contracts before launch; check64 remains unopened, no observed-check tuning/adoption/TEST/seed resume.'

grounding_current_admission_path=root/'reports/VG_GROUNDING_HUMAN_STATUS_RESOURCE_20261102.json'
if grounding_current_admission_path.exists() and 'grounding_controls_preparation' in state:
    admission=json.loads(grounding_current_admission_path.read_text(encoding='utf8'))
    state['grounding_controls_preparation']['latest_direct_human_resource_snapshot']=admission
    state['grounding_controls_preparation']['Qwen_GPU_currently_admitted']=admission['admitted']
    state['grounding_controls_preparation']['implementation_can_continue_without_GPU']=True

grounding_R1_launch_path=root/'reports/VG_GROUNDING_TRAIN_DEV_R1_LAUNCH_20261103.json'
if grounding_R1_launch_path.exists():
    launch=json.loads(grounding_R1_launch_path.read_text(encoding='utf8'))
    if launch.get('status')=='LAUNCHED':
        p=root/'reports/VG_GROUNDING_TRAIN_DEV_R1_LAST_STATUS_20261103.json'
        current=json.loads(p.read_text(encoding='utf8')) if p.exists() else {}
        active=current.get('status.json',{})
        status=active.get('status','LAUNCHED')
        state['grounding_TRAIN_dev_R1']={'status':status,'launch':launch,'last_snapshot':current,'controller_pid':launch['controller_pid'],'source_manifest':'configs/vg_grounding_train_dev_source_manifest_r1_20261103.json','config':'configs/vg_grounding_train_dev_r1_20261103.json','preregistration':'research/VG_GROUNDING_TRAIN_DEV_R1_PREREGISTRATION_20261103.md','main_GPU_experiment_fully_implemented':True,'maximum_new_Qwen_calls':1312,'G0_TRAIN_reused_not_regenerated':True,'check_or_TEST_opened':False,'policy_adopted':False,'automatic_retry':False}
        state['grounding_controls_preparation'].update(main_GPU_experiment_fully_implemented=True,status='CONSUMED_BY_FROZEN_TRAIN_DEV_R1_DO_NOT_RERUN_PREPARATION')
        state['program_status']='GROUNDING_TRAIN_DEV_R1_'+status
        state['current_active_compute']='vg_grounding_train_dev_r1_20261103' if status in ['RUNNING','LAUNCHED'] else None
        state['whole_program_complete']=False
        state['next_independent_actions'][0]='At next due12h read compact R1active status once via remote_vg_grounding_status_r1_20261103.py, no resource polling/relaunch. On new completion fetch returned grounding_train_dev_results.zip and rawSHAverify source/prereg/G0TRAINreuse/2400TRAIN+768dev records/3x800+3x256native corpora/dev-only selection before conclusion. On failure preserve evidence/no automatic retry. Fresh check/TEST and seeds remain closed.'

grounding_R1_verified_path=root/'reports/VG_GROUNDING_TRAIN_DEV_R1_COMPLETION_VERIFICATION_20261103.json'
grounding_R1_conclusion_path=root/'reports/VG_GROUNDING_TRAIN_DEV_R1_CONCLUSION_20261103.json'
if grounding_R1_verified_path.exists() and grounding_R1_conclusion_path.exists():
    verified=json.loads(grounding_R1_verified_path.read_text(encoding='utf8'));conclusion=json.loads(grounding_R1_conclusion_path.read_text(encoding='utf8'))
    assert verified['status']=='PASS_R1_SOURCE_PREREG_INPUT_G0_REUSE_AND3168_TRAIN_DEV_RECORDS_SIX_NATIVE_CORPORA' and verified['TRAIN_records']==2400 and verified['dev_records']==768 and verified['selected_arm_from_dev']=='G0'
    state['grounding_TRAIN_dev_R1'].update(status='COMPLETE_VERIFIED_DEV_SELECTED_G0_NO_POLICY_ADOPTION',completion_verification=verified,conclusion=conclusion,terminal_phase_polling_allowed=False,current_active_compute=None,automatic_check_or_TEST_launch=False)
    state['program_status']='GROUNDING_TRAIN_DEV_R1_COMPLETE_G0_SELECTED_NEW_GATE_PREPARATION_PENDING'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Preserve dev-selectedG0baseline and review pre-fixed20dev support packet as unblinded text audit(no human/visual accuracy) if useful. Prepare a separate heldout gate or finite TRAIN/dev fewshot/color amendment; no automatic check/TEST, duplicateG0-vsG0generation, observed-check tuning/adoption or seedresume. Completed grounding phase is terminal: no polling, model or metric rerun.'

grounding_review20_complete_path=root/'reports/VG_GROUNDING_DEV_REVIEW20_COMPLETION_20261104.json'
if grounding_review20_complete_path.exists():
    done=json.loads(grounding_review20_complete_path.read_text(encoding='utf8'))
    assert done['status']=='COMPLETE_ORIGINAL_IMAGE_FIXED20_DEV_REVIEW_PACKET_VERIFIED_PENDING_ANNOTATION' and done['units']==20 and done['source_images']==16 and done['human_labels_created']==0
    state['grounding_fixed20_original_image_review_packet']={'status':done['status'],'completion':done,'entrypoint':done['local_entrypoint'],'conclusion':'research/VG_GROUNDING_DEV_REVIEW20_CONCLUSION_20261104.md','review_completed':False,'current_active_compute':None,'policy_adopted':False}
    state['program_status']='GROUNDING_FIXED20_IMAGE_REVIEW_PACKET_READY_FEWSHOT_PREPARATION_PENDING'
    state['current_active_compute']=None
    state['whole_program_complete']=False
    state['next_independent_actions'][0]='Review fixed20already-observedDEV image/SG/GT/G0packet with explicit provenance; blank annotation is not human labels, unblinded assistant review not visual accuracy. Prepare finite TRAIN/dev zero/staticK2/dynamicK2 fewshot amendment with TRAIN-only color-blind examples/sharedfrozenSG/nativebudgets, exactG0reuse, no prompt tuning from check/TEST. Separatecolorablation later; scientific prereg/contracts/admission before any new generation. No automatic freshcheck/TEST/adoption/seedresume.'

path.write_text(json.dumps(state, indent=2), encoding='utf8')
print(json.dumps({'program_state_updated': True, 'locked_v2_local_source_hashes': 'PASS',
                  'caption_test_scored': False}))
