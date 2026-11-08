"""Remove one staged byte-identical CLIP copy under the prior explicit authorization."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os
home=Path.home().resolve();source=home/'coco_kb_v3/incoming/clip-vit-base-patch32-pytorch_model.bin';canonical=home/'shared-assets/models/clip-vit-base-patch32/pytorch_model.bin'
assert source.parent.resolve().is_relative_to(home) and canonical.resolve().is_relative_to(home)
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
expected='a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f';assert sha(canonical)==expected
before=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
if source.is_symlink():assert source.resolve()==canonical;action='already_alias'
else:
    assert source.is_file() and source.stat().st_uid==os.getuid() and sha(source)==expected
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid() or int(proc.name)==os.getpid():continue
            for fd in (proc/'fd').iterdir():
                try:assert os.readlink(fd)!=str(source),'active_consumer'
                except OSError:pass
            assert str(source) not in (proc/'maps').read_text(),'active_mapped_consumer'
        except OSError:pass
    source.unlink();source.symlink_to(canonical);assert source.resolve()==canonical;action='duplicate_replaced_with_alias'
result={'at_utc':datetime.now(timezone.utc).isoformat(),'source':'${REMOTE_HOME}/coco_kb_v3/incoming/clip-vit-base-patch32-pytorch_model.bin','canonical':'${REMOTE_HOME}/shared-assets/models/clip-vit-base-patch32/pytorch_model.bin','sha256':expected,'action':action,'bytes':canonical.stat().st_size,'observed_free_gain_bytes':os.statvfs(home).f_bavail*os.statvfs(home).f_frsize-before,'unique_old_project_results_deleted':False}
(home/'shared-assets/maintenance/STAGED_CLIP_CONSOLIDATION_20261010.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
