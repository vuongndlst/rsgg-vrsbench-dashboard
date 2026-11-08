"""Low-priority directory-only storage inventory; no deletion or job changes."""
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, io, json, os, re, shutil, subprocess, time

started = time.monotonic()
home = Path.home()
root = home / 'rsg-former'
os.nice(19)
if hasattr(os, 'sched_setaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})

def alias(path):
    return re.sub(r'/(home|users)/[^/\s]+', r'/\1/${ACCOUNT}',
                  str(path).replace(str(home), '${REMOTE_HOME}'))

def capacity(path):
    stat = os.statvfs(path)
    return {'path': alias(path), 'total_bytes': stat.f_blocks * stat.f_frsize,
            'used_bytes': (stat.f_blocks - stat.f_bfree) * stat.f_frsize,
            'free_bytes': stat.f_bfree * stat.f_frsize,
            'available_bytes': stat.f_bavail * stat.f_frsize,
            'reserved_or_unavailable_bytes': (stat.f_bfree - stat.f_bavail) * stat.f_frsize}

def scan(path, depth=None):
    args = ['du', '-x', '-B1', '--null']
    if depth is not None:
        args.append('--max-depth=' + str(depth))
    if path == Path('/'):
        args.extend('--exclude=' + item for item in ('/proc', '/sys', '/dev', '/run'))
    args.append(str(path))
    if shutil.which('ionice'):
        args = ['ionice', '-c', '3'] + args
    begin = time.monotonic()
    try:
        proc = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=120)
        raw, errors, code = proc.stdout, proc.stderr, proc.returncode
    except subprocess.TimeoutExpired as error:
        raw, errors, code = error.stdout or b'', error.stderr or b'', 'timeout'
    rows = []
    for record in raw.split(b'\0'):
        if not record:
            continue
        amount, separator, name = record.partition(b'\t')
        if separator and amount.isdigit():
            rows.append({'path': alias(os.fsdecode(name)), 'allocated_bytes': int(amount)})
    return {'scope': alias(path), 'max_depth': depth, 'returncode': code,
            'stderr_lines': len(errors.splitlines()),
            'complete_without_reported_errors': code == 0,
            'elapsed_seconds': time.monotonic() - begin,
            'rows': sorted(rows, key=lambda row: (-row['allocated_bytes'], row['path']))}

before = capacity(home)
mounts, seen = [], set()
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    first, second = line.split(' - ', 1)
    target = first.split()[4].replace('\\040', ' ')
    kind = second.split()[0]
    if kind not in ('ext4', 'xfs', 'btrfs', 'zfs', 'nfs', 'nfs4', 'overlay', 'fuseblk'):
        continue
    try:
        device = os.stat(target).st_dev
        if device in seen:
            continue
        seen.add(device)
        mounts.append(dict(capacity(target), filesystem_type=kind))
    except OSError:
        pass
root_scan = scan(Path('/'), 1)
home_scan = scan(home)
row_map = {row['path']: row['allocated_bytes'] for row in home_scan['rows']}
entries = []
for path in sorted(home.iterdir()):
    try:
        if not path.is_dir() and not path.is_symlink():
            continue
        item = {'path': alias(path), 'symlink': path.is_symlink(),
                'allocated_bytes': row_map.get(alias(path)), 'directory': path.is_dir()}
        if path.is_symlink():
            item['link_target'] = alias(os.readlink(path))
            item['allocated_bytes'] = path.lstat().st_blocks * 512
        entries.append(item)
    except OSError:
        entries.append({'path': alias(path), 'unreadable': True, 'allocated_bytes': None})
entries.sort(key=lambda row: (-(row.get('allocated_bytes') or 0), row['path']))
after = capacity(home)
full = {'utc': datetime.now(timezone.utc).isoformat(), 'mode': 'directory_only_read_only_scan',
        'filesystems': mounts, 'home_capacity_before': before, 'home_capacity_after': after,
        'root_directories': root_scan, 'home_all_directories': home_scan,
        'home_top_level_entries': entries,
        'measurement': 'du allocated filesystem blocks, physical directories; no following directory symlinks; one filesystem per scan; hard links counted once per invocation',
        'limits': ['Root lists depth1 only and excludes virtual /proc,/sys,/dev,/run',
                   'All physical home directories are included when the home scan finishes without errors',
                   'Permission errors and mounted filesystems limit coverage; no privilege bypass',
                   'Parent sizes include descendants: do not sum all rows',
                   'Files directly under a parent, deleted-but-open files, reserved space and concurrent changes can explain df/du differences'],
        'jobs_modified': False, 'assets_deleted': False, 'cpu_affinity_count': 1,
        'wall_seconds': time.monotonic() - started}
folder = root / 'reports'
folder.mkdir(exist_ok=True)
raw = json.dumps(full, ensure_ascii=False, indent=2).encode('utf8')
(folder / 'SERVER_DIRECTORY_SIZES_FULL.json').write_bytes(raw)
csv_buffer = io.StringIO(newline='')
writer = csv.writer(csv_buffer)
writer.writerow(['scope', 'path', 'allocated_bytes', 'GiB'])
for scope, scan_result in [('root_depth1', root_scan), ('home_recursive', home_scan)]:
    for row in scan_result['rows']:
        writer.writerow([scope, row['path'], row['allocated_bytes'], round(row['allocated_bytes'] / 2**30, 6)])
(folder / 'SERVER_DIRECTORY_SIZES_ALL.csv').write_text(csv_buffer.getvalue(), encoding='utf-8-sig')
summary = {key: full[key] for key in ('utc', 'filesystems', 'home_capacity_before', 'home_capacity_after',
                                   'home_top_level_entries', 'measurement', 'limits', 'jobs_modified', 'assets_deleted', 'wall_seconds')}
summary['root_directories'] = root_scan
summary['home_scan'] = {key: value for key, value in home_scan.items() if key != 'rows'}
summary['home_directory_count'] = len(home_scan['rows'])
summary['home_total_allocated_bytes'] = row_map.get('${REMOTE_HOME}')
summary['largest_home_descendants'] = [row for row in home_scan['rows'] if row['path'].count('/') >= 2][:30]
summary['full_json_bytes'] = len(raw)
summary['full_json_sha256'] = hashlib.sha256(raw).hexdigest()
print(json.dumps(summary, ensure_ascii=False))
