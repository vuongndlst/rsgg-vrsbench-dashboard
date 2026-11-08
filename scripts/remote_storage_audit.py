"""Inspect writable storage capacities without returning device/network mount sources."""
import os,json,time
from pathlib import Path
out={'utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'filesystems':[]};seen=set()
for line in Path('/proc/self/mountinfo').read_text().splitlines():
    before,after=line.split(' - ',1);fields=before.split();kind=after.split()[0];target=fields[4].replace('\\040',' ')
    if kind not in ['ext4','xfs','btrfs','zfs','nfs','nfs4','overlay','fuseblk']:continue
    try:
        stat=os.stat(target);v=os.statvfs(target)
        if stat.st_dev in seen:continue
        seen.add(stat.st_dev)
        out['filesystems'].append({'mount':target,'filesystem_type':kind,'total_bytes':v.f_blocks*v.f_frsize,'available_bytes':v.f_bavail*v.f_frsize,'account_write_access':os.access(target,os.W_OK)})
    except OSError:pass
out['star_archive_required_bytes']=120735953354
root=Path.home()/'rsg-former'
out['project_directory_write_access']=os.access(root,os.W_OK)
out['project_filesystem_available_bytes']=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
# W_OK on a mount root is not W_OK in the user's project directory on that mount.
out['project_storage_fits_STAR_plus_20percent_reserve']=out['project_directory_write_access'] and out['project_filesystem_available_bytes']>out['star_archive_required_bytes']*1.2
out['alternative_writable_mount_with_space']=any(i['mount']!='/' and i['account_write_access'] and i['available_bytes']>out['star_archive_required_bytes']*1.2 for i in out['filesystems'])
(root/'reports/P0_STORAGE_AUDIT.json').write_text(json.dumps(out,indent=2));print(json.dumps(out))
