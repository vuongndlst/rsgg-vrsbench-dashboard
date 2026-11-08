"""Read-only diagnostic of this isolated full-stream fixture, not VG jobs."""
from pathlib import Path
import json, sqlite3, datetime
root = Path.home() / 'rsg-former'
folder = root / 'artifacts/caption_readiness/oi_full_stream_ledger_fixture_v1'
out = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(), 'fixture_exists':folder.exists(), 'cases':[], 'writer_processes':[]}
if folder.exists():
    for path in sorted(folder.glob('*/artifacts/caption_readiness/*.sqlite')):
        db = sqlite3.connect('file:'+str(path)+'?mode=ro', uri=True, timeout=2)
        try:
            out['cases'].append({'case':path.parents[2].name,
                'pixels':db.execute('SELECT COUNT(*) FROM pixels').fetchone()[0],
                'attempts':db.execute('SELECT COUNT(*) FROM attempts').fetchone()[0]})
        except sqlite3.OperationalError as error:
            out['cases'].append({'case':path.parents[2].name, 'error':str(error)})
        finally:
            db.close()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            matches = []
            for fd in (proc/'fd').iterdir():
                try:
                    target = str(fd.readlink())
                    if str(folder) in target: matches.append(target.replace(str(folder), '${FIXTURE}'))
                except OSError: pass
            if matches:
                out['writer_processes'].append({'pid':int(proc.name), 'fixture_fds':matches, 'state':(proc/'status').read_text().splitlines()[2]})
        except (OSError, PermissionError): pass
    out['outcome_exists'] = (folder/'outcome.json').exists()
print(json.dumps(out))
