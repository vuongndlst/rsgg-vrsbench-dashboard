"""Stop only the identified owned synthetic fixture after its wall overrun."""
from pathlib import Path
from datetime import datetime, timezone
import os, signal, json, hashlib, time
root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness/oi_full_stream_ledger_fixture_v1'
pid = 1090355
proc = Path('/proc') / str(pid)
out = {'utc':datetime.now(timezone.utc).isoformat(), 'pid':pid,
       'reason':'synthetic fixture total120s alarm overwritten by each production batch100s alarm; disk commit latency made total cap ineffective',
       'fixture_source_sha256':hashlib.sha256((root/'scripts/remote_oi_full_stream_ledger_probe.py').read_bytes()).hexdigest(),
       'production_source_sha256':hashlib.sha256((root/'scripts/remote_oi_stream_fingerprint_batch.py').read_bytes()).hexdigest(),
       'live_v2_modified':False, 'real_pixel_journals_modified':False, 'network_downloaded_bytes':0}
if proc.exists():
    command = (proc/'cmdline').read_bytes()
    assert b'python' in command and b'-B\x00-\x00' in command, 'unexpected_fixture_command'
    fds = []
    for fd in (proc/'fd').iterdir():
        try:
            if str(folder) in str(fd.readlink()): fds.append(str(fd.readlink()).replace(str(folder),'${FIXTURE}'))
        except OSError: pass
    assert any('failure_retry_guard' in fd for fd in fds), 'fixture_ownership_not_proven'
    fields = (proc/'stat').read_text().split(') ',1)[1].split()
    elapsed = float(Path('/proc/uptime').read_text().split()[0]) - int(fields[19])/os.sysconf('SC_CLK_TCK')
    assert elapsed > 120
    out.update(elapsed_seconds_before_stop=elapsed, verified_fixture_fds=fds)
    os.kill(pid, signal.SIGTERM)
    time.sleep(1)
    out['signal_sent'] = 'SIGTERM'
    out['process_path_remaining'] = proc.exists()
else:
    out['signal_sent'] = None
    out['already_exited'] = True
(folder/'stopped_overrun.json').write_text(json.dumps(out,indent=2))
(root/'reports/OI_FULL_STREAM_LEDGER_INITIAL_FAILURE.json').write_text(json.dumps(out,indent=2))
print(json.dumps(out))
