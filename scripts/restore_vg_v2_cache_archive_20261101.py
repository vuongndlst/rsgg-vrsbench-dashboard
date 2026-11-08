"""Restore byte-exact original .pt files from the verified local VGBS4GZ1 archive."""
from pathlib import Path
import argparse,json,hashlib,gzip,struct,shutil,os
import numpy as np
p=argparse.ArgumentParser();p.add_argument('archive_directory');p.add_argument('destination');a=p.parse_args();base=Path(a.archive_directory).resolve();dest=Path(a.destination).resolve();receipt=json.loads((base/'backup_verification.json').read_text());assert receipt['status']=='PASS_ALL981_LOCAL_ARCHIVES_RESTORED_TO_ORIGINAL_SHA256'
assert shutil.disk_usage(dest.parent).free>receipt['raw_bytes']+1024**3;dest.mkdir(exist_ok=False)
for row in receipt['rows']:
 name=row['name'];assert Path(name).name==name
 encoded=(base/(name+'.bs4.gz')).read_bytes();assert hashlib.sha256(encoded).hexdigest()==row['archive_sha256'] and encoded[:8]==b'VGBS4GZ1'
 n=struct.unpack('<Q',encoded[8:16])[0];assert n==row['raw_bytes'];body=gzip.decompress(encoded[16:]);assert len(body)==n;end=n-n%4
 raw=np.frombuffer(body[:end],dtype=np.uint8).reshape(4,-1).T.copy().tobytes()+body[end:];assert hashlib.sha256(raw).hexdigest()==row['original_sha256']
 with (dest/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
manifest=(base/'manifest.json').read_bytes();assert hashlib.sha256(manifest).hexdigest()==receipt['cache_manifest_sha256'];(dest/'manifest.json').write_bytes(manifest)
print(json.dumps({'status':'RESTORED_ALL981_RAW_SHARDS_ORIGINAL_SHA256','destination':str(dest),'manifest_sha256':receipt['cache_manifest_sha256']}))
