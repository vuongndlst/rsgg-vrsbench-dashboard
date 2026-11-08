"""Read-only preflight for five user-named legacy folders and /tmp."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, subprocess, time, stat

HOME = Path.home().resolve()
PROJECT = HOME / 'rsg-former'
NAMES = ('coco_kb_v3', 'v2_kb_20260909', 'v2_kb', 'image-captioning-kg', 'sgk_ocr')
TARGETS = [HOME / n for n in NAMES]
CORE = {
    'src/rsgformer/vg_v2_runtime.py': '0d7157491c093a02362f25c8841aab50a4b95791d9de3f08550ee37faff6aff8',
    'scripts/remote_vg_v2_pipeline.py': 'b336c5a84f5a50eb8e667015efda8f619ae4def3129f84007cc8b3b8fb2e59aa',
    'scripts/remote_vg_v2_probe.py': '4e58967bb2bd74bd8303a082360e68be726d226d8a7c016e4f291f7b90f5c560',
    'configs/vg_method_v2.json': '4ac00b7819dd37f1b894f7301737e6a3d663643dc02a6e63b8344a3c31322d68',
}

def label(p):
    return str(p).replace(str(HOME), '${REMOTE_HOME}')

def sha(p):
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def within(value, root):
    value = value.removesuffix(' (deleted)')
    return value == str(root) or value.startswith(str(root) + '/')

def available():
    s = os.statvfs(HOME)
    return {'available_bytes': s.f_bavail * s.f_frsize,
            'free_blocks_bytes': s.f_bfree * s.f_frsize}

def mounts_under(root):
    found = []
    for line in Path('/proc/self/mountinfo').read_text().splitlines():
        parts = line.split()
        p = parts[4].replace('\\040', ' ').replace('\\011', '\t').replace('\\134', '\\')
        if within(p, root):
            found.append(label(p))
    return found

def validate_target(p):
    assert p in TARGETS and p.parent == HOME and p.name in NAMES
    assert p.is_absolute() and p != HOME and p.parent.resolve() == HOME
    if not os.path.lexists(p):
        return {'name': p.name, 'path': label(p), 'present': False}
    s = p.lstat()
    assert s.st_uid == os.getuid(), 'target_not_owned:' + p.name
    assert stat.S_ISDIR(s.st_mode) and not p.is_symlink(), 'target_not_physical_directory:' + p.name
    assert p.resolve() == p and not mounts_under(p), 'target_mount_or_redirect:' + p.name
    return {'name': p.name, 'path': label(p), 'present': True,
            'identity': [s.st_dev, s.st_ino], 'uid': s.st_uid}

def consumers(roots):
    refs, inaccessible = [], 0
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name) == os.getpid():
            continue
        try:
            uid = proc.stat().st_uid
            cmd = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            comm = (proc / 'comm').read_text().strip()[:80]
        except OSError:
            continue
        paths = []
        for name in ('cwd', 'exe', 'root'):
            try:
                paths.append((name, os.readlink(proc / name)))
            except OSError:
                pass
        try:
            for fd in (proc / 'fd').iterdir():
                try:
                    paths.append(('fd', os.readlink(fd)))
                except OSError:
                    pass
        except PermissionError:
            inaccessible += 1
        except OSError:
            pass
        try:
            for line in (proc / 'maps').read_text().splitlines():
                fields = line.split(maxsplit=5)
                if len(fields) == 6 and fields[5].startswith('/'):
                    paths.append(('mapped_file', fields[5]))
        except OSError:
            pass
        for root in roots:
            matches = {(kind, value) for kind, value in paths if within(value, root)}
            # Absolute path arguments conservatively block deletion too.
            if str(root) in cmd:
                matches.add(('argument', str(root)))
            for kind, value in sorted(matches):
                refs.append({'pid': int(proc.name), 'uid': uid, 'comm': comm,
                             'root': label(root), 'kind': kind, 'path': label(value)})
    return {'references': refs, 'fd_permission_denied_processes': inaccessible,
            'limit': 'Other-user /proc details may be inaccessible; no process signals.'}

def inbound_links():
    matches, errors = [], []
    # Protect running research and shared assets; old unrelated home trees are outside this audit.
    start = time.monotonic()
    protected_roots = [HOME / name for name in ('shared-assets', 'rsg-former', 'skgcap_vlm', 'sbir', 'fair-project')]
    pending = [p for p in protected_roots if p.is_dir() and not p.is_symlink()]
    while pending:
        if time.monotonic() - start > 75:
            raise RuntimeError('inbound_link_scan_time_limit')
        parent = pending.pop()
        try:
            with os.scandir(parent) as entries:
                for entry in entries:
                    if parent == HOME and (entry.name in NAMES or entry.name == '.ssh'):
                        continue
                    # DirEntry uses directory type information, avoiding stat on every file.
                    if entry.is_symlink():
                        p = Path(entry.path)
                        try:
                            resolved = str(p.resolve())
                        except (OSError, RuntimeError):
                            continue
                        for target in TARGETS:
                            if within(resolved, target):
                                matches.append({'link': label(p), 'target': label(resolved),
                                                'legacy_root': target.name})
                    elif entry.is_dir(follow_symlinks=False):
                        pending.append(Path(entry.path))
        except OSError as e:
            errors.append(label(e.filename))
    return {'inbound': matches, 'scan_errors': errors,
            'scanned_roots': [label(p) for p in protected_roots],
            'limit': 'Protected research, shared datasets/models, retained caption project and fair-project; unrelated old projects not scanned.'}

def protected_snapshot():
    result = {'core_sha256': {}, 'stat_records': {}, 'shared_tree_digest': None}
    for rel, expected in CORE.items():
        digest = sha(PROJECT / rel)
        assert digest == expected, 'locked_core_changed:' + rel
        result['core_sha256'][rel] = digest
    protected = [HOME / 'skgcap_vlm', HOME / 'shared-assets',
                 HOME / 'sbir/hienlth/vg/images.zip', HOME / 'sbir/hienlth/vg/images2.zip']
    for p in protected:
        s = p.stat()
        result['stat_records'][label(p)] = [s.st_dev, s.st_ino, s.st_size]
    # Shared models, metadata and physical images are immutable in this cleanup.
    rows = []
    shared = HOME / 'shared-assets'
    for current, dirs, files in os.walk(shared, followlinks=False):
        dirs.sort()
        files.sort()
        if Path(current) == shared:
            dirs[:] = [d for d in dirs if d != 'maintenance']
        for name in dirs + files:
            p = Path(current) / name
            s = p.lstat()
            rows.append([str(p.relative_to(shared)), s.st_mode, s.st_dev, s.st_ino,
                         s.st_size, s.st_mtime_ns, os.readlink(p) if p.is_symlink() else None])
    result['shared_tree_digest'] = hashlib.sha256(json.dumps(sorted(rows)).encode()).hexdigest()
    result['shared_tree_entries'] = len(rows)
    return result

def du(paths, depth=0):
    cmd = ['du', '-x', '-B1', '--null', '--max-depth=' + str(depth), '--'] + [str(p) for p in paths]
    if Path('/usr/bin/ionice').exists():
        cmd = ['ionice', '-c3'] + cmd
    completed = subprocess.run(cmd, capture_output=True, timeout=90)
    rows = []
    for record in completed.stdout.split(b'\0'):
        if record:
            size, path = record.decode(errors='replace').split('\t', 1)
            rows.append({'path': label(path), 'allocated_bytes': int(size)})
    return {'returncode': completed.returncode, 'rows': sorted(rows, key=lambda x: -x['allocated_bytes']),
            'stderr_lines': len(completed.stderr.splitlines())}

def run():
    os.nice(19)
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    roots = [validate_target(p) for p in TARGETS]
    process = consumers(TARGETS + [Path('/tmp')])
    links = inbound_links()
    legacy_refs = [r for r in process['references'] if r['root'] != '/tmp']
    tmp = du([Path('/tmp')], depth=2)
    top = []
    for p in Path('/tmp').iterdir():
        try:
            s = p.lstat()
            top.append({'path': str(p), 'uid': s.st_uid, 'directory': p.is_dir(),
                        'symlink': p.is_symlink(), 'mtime_utc': datetime.fromtimestamp(s.st_mtime, timezone.utc).isoformat(),
                        'logical_bytes_if_file': s.st_size if stat.S_ISREG(s.st_mode) else None})
        except OSError:
            continue
    report = {'at_utc': datetime.now(timezone.utc).isoformat(), 'mode': 'read_only_preflight',
              'deletion_names': list(NAMES), 'targets': roots,
              'target_sizes': du([p for p in TARGETS if p.exists()]),
              'process_scan': process, 'links': links,
              'protected_snapshot': protected_snapshot(), 'disk': available(),
              'tmp_sizes_depth2': tmp, 'tmp_top_entries': top,
              'ready_for_explicitly_authorized_five_deletions': not legacy_refs and not links['inbound'] and not links['scan_errors'],
              'tmp_deletion_authorized': False, 'deleted': False}
    dest = PROJECT / 'reports/LEGACY_DELETE_PREFLIGHT_20261011.json'
    dest.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))

if __name__ == '__main__':
    run()
