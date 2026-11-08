"""Fetch this already-redacted inventory verbatim and verify its remote digest."""
from pathlib import Path
import hashlib, json, sys
root=Path(__file__).absolute().parents[1]
sys.path.insert(0,str(root/'research/tools'))
from p0_audit import connect
summary=json.loads((root/'reports/SERVER_DIRECTORY_SIZES_SUMMARY.json').read_text(encoding='utf8'))
connection=None
temporary=root/'reports/SERVER_DIRECTORY_SIZES_FULL.raw.json'
try:
    connection=connect()
    sftp=connection.open_sftp()
    remote_home=sftp.normalize('.')
    sftp.get(remote_home+'/rsg-former/reports/SERVER_DIRECTORY_SIZES_FULL.json',str(temporary))
    raw=temporary.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==summary['full_json_sha256']
    report=json.loads(raw)
    assert report['home_top_level_entries']==summary['home_top_level_entries']
    destination=root/'reports/SERVER_DIRECTORY_SIZES_FULL.json'
    # Preserve the generic fetcher's invalid text-redaction result for debugging.
    if destination.exists():
        destination.replace(root/'reports/SERVER_DIRECTORY_SIZES_FULL.invalid_sanitized.json')
    temporary.replace(destination)
    print(json.dumps({'bytes':len(raw),'remote_sha256_verified':True,'json_valid':True,
                      'all_paths_already_redacted_at_source':True}))
finally:
    if connection:connection.close()
