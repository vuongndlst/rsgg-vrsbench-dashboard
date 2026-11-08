"""Compact, deduplicated monitor for the human-amended single component test."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).absolute().parents[1]


def completion_event(snapshot):
    result = snapshot.get('VG_COMPONENT_CONTROL_TEST_RESULT.json', {})
    status = result.get('status')
    if status == 'COMPONENT_TEST_COMPLETE':
        assert result['test_scored'] and result['test_passes'] == 1
        return {'type': status, 'checkpoint_sha256': result['checkpoint_sha256'],
                'selected_epoch': result['selected_epoch'], 'seed': result['seed'],
                'test': result['test'], 'comparability_limitations': result['comparability_limitations']}
    if status == 'COMPONENT_TEST_BLOCKED':
        return {'type': status, 'reason': result['reason'],
                'test_opened': result['test_opened'], 'no_blind_retry': True}
    if snapshot.get('phase_exists') and snapshot.get('process', {}).get('active') is False:
        return {'type': 'COMPONENT_PROCESS_EXITED_WITHOUT_RESULT',
                'pid': snapshot['process']['pid'], 'test_opened': snapshot['test_opened'],
                'no_blind_retry': True}
    return None


def atomic(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf8')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--record-only', action='store_true')
    args = parser.parse_args()
    status_path = ROOT / 'reports/VG_COMPONENT_CONTROL_TEST_LAST_STATUS.json'
    if not args.record_only:
        call = subprocess.run([sys.executable, '-X', 'utf8', str(ROOT/'scripts/ssh_project.py'),
                               'run', 'scripts/remote_vg_component_control_status.py',
                               '--environment', 'system', '--report', str(status_path)],
                              cwd=ROOT, capture_output=True, text=True, encoding='utf8')
        if call.returncode:
            # A transport failure does not prove that the remote worker stopped.
            raise RuntimeError('component_status_unavailable_remote_job_state_unknown')
    snapshot = json.loads(status_path.read_text(encoding='utf8'))
    assert snapshot['phase_id'] == 'vg_component_control_test_v1'
    notify_path = ROOT / 'reports/VG_COMPONENT_CONTROL_TEST_NOTIFY_STATE.json'
    state = json.loads(notify_path.read_text(encoding='utf8')) if notify_path.exists() else {'seen': []}
    event = completion_event(snapshot)
    key = hashlib.sha256(json.dumps(event, sort_keys=True).encode()).hexdigest() if event else None
    fresh = event is not None and key not in state['seen']
    if fresh:
        state['seen'].append(key)
        state['last_event'] = event
    state['updated_utc'] = datetime.now(timezone.utc).isoformat()
    atomic(notify_path, state)
    summary = {'status': 'checked', 'utc': snapshot['utc'], 'notify': fresh,
               'events': [event] if fresh else [], 'process': snapshot.get('process'),
               'test_opened': snapshot.get('test_opened'),
               'progress': snapshot.get('VG_COMPONENT_CONTROL_TEST_PROGRESS.json'),
               'result_status': snapshot.get('VG_COMPONENT_CONTROL_TEST_RESULT.json', {}).get('status')}
    atomic(ROOT/'reports/VG_COMPONENT_CONTROL_TEST_MONITOR_LAST_CHECK.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
