"""Preserve a YOLO weight referenced by the retained skgcap_vlm project before cleanup."""
from pathlib import Path
from datetime import datetime, timezone
import os, json, hashlib, shutil, tempfile

home = Path.home().resolve()
src = home / 'image-captioning-kg/yolov8x.pt'
link = home / 'skgcap_vlm/kgproj/yolov8x.pt'
dst = home / 'shared-assets/models/ultralytics/yolov8x.pt'
assert src.resolve() == src and src.is_file() and not src.is_symlink()
assert src.stat().st_uid == os.getuid()
assert link.is_symlink() and link.resolve() == src and link.lstat().st_uid == os.getuid()
assert src.is_relative_to(home / 'image-captioning-kg') and link.parent.resolve().is_relative_to(home / 'skgcap_vlm')
assert dst.parent.parent.resolve() == home / 'shared-assets/models'
assert not os.path.lexists(dst), 'shared_destination_already_exists'

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

os.nice(19)
digest = sha(src)
dst.parent.mkdir(exist_ok=True)
assert dst.parent.resolve() == dst.parent and dst.parent.stat().st_uid == os.getuid()
# Preserve first; no model deserialization or inference.
fd, temporary = tempfile.mkstemp(prefix='.yolo-preserve-', dir=dst.parent)
os.close(fd)
temporary = Path(temporary)
try:
    shutil.copyfile(src, temporary)
    assert sha(temporary) == digest and temporary.stat().st_size == src.stat().st_size
    # Hardlink publication requires an absent destination, never overwrites assets.
    os.link(temporary, dst)
finally:
    temporary.unlink(missing_ok=True)
temporary_link = link.parent / '.yolov8x.cleanup-20261011-link'
assert not os.path.lexists(temporary_link)
temporary_link.symlink_to(dst)
os.replace(temporary_link, link)
assert link.resolve() == dst and sha(link) == digest and sha(src) == digest
result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'RETAINED_DEPENDENCY_PRESERVED',
          'source': '${REMOTE_HOME}/image-captioning-kg/yolov8x.pt',
          'shared': '${REMOTE_HOME}/shared-assets/models/ultralytics/yolov8x.pt',
          'retained_link': '${REMOTE_HOME}/skgcap_vlm/kgproj/yolov8x.pt',
          'bytes': dst.stat().st_size, 'sha256': digest, 'link_resolves_to_verified_copy': True,
          'source_deleted': False, 'processes_signalled': False, 'model_loaded': False}
(home / 'rsg-former/reports/LEGACY_YOLO_PRESERVATION_20261011.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
