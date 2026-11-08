"""Bounded read-only discovery of Java runtimes for caption scorers."""
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import signal

signal.alarm(60)
home = Path.home()
roots = [Path('/usr/lib/jvm'), Path('/usr/java'), Path('/opt'), Path('/usr/local'),
         home / 'shared-assets', home / '.local', home / 'fair-project',
         home / 'etechs', home / 'rsg-former/.tools']
pending = deque((root, 0) for root in roots)
visited = set()
candidates = set()
on_path = shutil.which('java')
if on_path:
    candidates.add(Path(on_path))
while pending and len(visited) < 350:
    folder, depth = pending.popleft()
    if str(folder) in visited:
        continue
    visited.add(str(folder))
    possible = folder / 'bin/java'
    if possible.is_file() and os.access(possible, os.X_OK):
        candidates.add(possible)
    if folder.name == 'bin':
        possible = folder / 'java'
        if possible.is_file() and os.access(possible, os.X_OK):
            candidates.add(possible)
    if depth >= 4:
        continue
    try:
        entries = list(folder.iterdir())[:150]
    except OSError:
        continue
    for entry in entries:
        if entry.name in ['.git', '.ssh', '.aws', 'datasets', 'models', 'weights',
                          'images', 'node_modules', 'site-packages', 'artifacts']:
            continue
        try:
            if entry.is_dir() and (depth < 1 or re.search(r'java|jdk|jre|runtime|bin|tools', entry.name, re.I)):
                pending.append((entry, depth + 1))
        except OSError:
            pass

results = []
for candidate in sorted(candidates):
    try:
        process = subprocess.run([str(candidate), '-version'], capture_output=True, text=True, timeout=10)
        lines = (process.stderr or process.stdout).splitlines()
        results.append({'path_alias': str(candidate).replace(str(home), '${REMOTE_HOME}'),
                        'exit_code': process.returncode, 'version_first_line': lines[0] if lines else None})
    except (OSError, subprocess.TimeoutExpired) as error:
        results.append({'path_alias': str(candidate).replace(str(home), '${REMOTE_HOME}'),
                        'error_type': type(error).__name__})
report = {'utc': datetime.now(timezone.utc).isoformat(), 'mode': 'bounded_read_only',
          'visited_directories': len(visited), 'pending_directories': len(pending),
          'depth_limit': 4, 'exhaustive_server_search': False, 'runtimes': results,
          'shared_environment_modified': False}
(home / 'rsg-former/reports/JAVA_RUNTIME_DISCOVERY.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report))
