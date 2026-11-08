"""Delete only the five home directories explicitly named by the user."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, runpy, shutil, time

home = Path.home().resolve()
project = home / 'rsg-former'
helper = project / 'scripts/remote_legacy_delete_preflight.py'
assert hashlib.sha256(helper.read_bytes()).hexdigest() == 'df7f1688d7ff7db60be6cf011b4b14614d0d858230bb833936910a5c6646a7af'
h = runpy.run_path(str(helper), run_name='cleanup_helper')
names = ('coco_kb_v3', 'v2_kb_20260909', 'v2_kb', 'image-captioning-kg', 'sgk_ocr')
assert h['NAMES'] == names
targets = [home / n for n in names]
preflight = json.loads((project / 'reports/LEGACY_DELETE_PREFLIGHT_20261011.json').read_text())
assert preflight['ready_for_explicitly_authorized_five_deletions'], 'preflight_not_ready'
assert preflight['deletion_names'] == list(names)
age = (datetime.now(timezone.utc) - datetime.fromisoformat(preflight['at_utc'])).total_seconds()
assert 0 <= age <= 1800, 'preflight_stale'
assert shutil.rmtree.avoids_symlink_attacks, 'require_fd_based_no_follow_rmtree'
os.nice(19)
os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

events = []
dest = project / 'reports/LEGACY_DELETE_RESULT_20261011.json'
protected_before = h['protected_snapshot']()
assert protected_before == preflight['protected_snapshot'], 'protected_assets_changed_since_preflight'
links = h['inbound_links']()
assert not links['inbound'] and not links['scan_errors'], 'inbound_links_or_scan_errors'
free_before = h['available']()
start = time.monotonic()

def save(status):
    dest.write_text(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                               'status': status, 'explicit_authorization_names': list(names),
                               'events': events, 'disk_before': free_before,
                               'tmp_deleted': False, 'processes_signalled': False}, indent=2))

for p in targets:
    current = h['validate_target'](p)
    original = next(r for r in preflight['targets'] if r['name'] == p.name)
    if not current['present']:
        events.append({'name': p.name, 'action': 'already_absent'})
        save('IN_PROGRESS')
        continue
    assert original['present'] and original['identity'] == current['identity'], 'target_identity_changed:' + p.name
    active = h['consumers']([p])['references']
    if active:
        events.append({'name': p.name, 'action': 'skipped_active_consumer', 'references': active})
        save('IN_PROGRESS')
        continue
    events.append({'name': p.name, 'path': h['label'](p), 'action': 'delete_prepared',
                   'identity': current['identity']})
    save('IN_PROGRESS')
    # Recursive removal is confined to this validated physical home directory.
    # CPython's fd-based rmtree unlinks symlinks rather than following them.
    shutil.rmtree(p)
    assert not os.path.lexists(p), 'target_still_present:' + p.name
    events.append({'name': p.name, 'action': 'removed', 'absent_verified': True})
    save('IN_PROGRESS')

protected_after = h['protected_snapshot']()
assert protected_after == protected_before, 'protected_asset_verification_failed'
disk_after = h['available']()
result = {'at_utc': datetime.now(timezone.utc).isoformat(),
          'status': 'COMPLETE' if all(not os.path.lexists(p) for p in targets) else 'PARTIAL_ACTIVE_CONSUMER',
          'events': events, 'explicit_authorization_names': list(names),
          'disk_before': free_before, 'disk_after': disk_after,
          'observed_available_gain_bytes': disk_after['available_bytes'] - free_before['available_bytes'],
          'previous_allocated_target_bytes': sum(r['allocated_bytes'] for r in preflight['target_sizes']['rows']),
          'elapsed_seconds': time.monotonic() - start,
          'protected_assets_unchanged': protected_after == protected_before,
          'protected_snapshot': protected_after, 'skgcap_vlm_preserved': (home / 'skgcap_vlm').is_dir(),
          'tmp_deleted': False, 'processes_signalled': False,
          'known_pipeline_pids_present': {str(pid): Path('/proc', str(pid)).exists() for pid in (3853348, 1113624, 648773)}}
dest.write_text(json.dumps(result, indent=2))
print(json.dumps(result))
