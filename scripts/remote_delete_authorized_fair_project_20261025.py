"""Delete only the exact user-authorized home/fair-project; never follow aliases."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, resource, shutil, signal, stat, subprocess, time

started=time.monotonic()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
os.nice(10)
resource.setrlimit(resource.RLIMIT_AS,(384*1024**2,384*1024**2))
signal.alarm(150)
home=Path.home().resolve()
target=home/'fair-project'
project=home/'rsg-former'
out={'utc':datetime.now(timezone.utc).isoformat(),'authorization':'Human explicit request2026-10-25: delete fair-project on server',
     'target':str(target),'files_outside_target_deleted':False,'processes_killed':False,
     'disk_free_before_bytes':shutil.disk_usage(project).free,'deletion_started':False}
try:
    assert target.name=='fair-project' and target.parent==home
    assert target!=home and target!=project and not target.is_symlink()
    if not target.exists():
        out['status']='ALREADY_ABSENT_NO_DELETION'
    else:
        assert target.resolve(strict=True)==target
        initial=target.stat()
        assert stat.S_ISDIR(initial.st_mode) and initial.st_uid==os.getuid()
        assert initial.st_dev==home.stat().st_dev
        assert not (project.resolve()==target or target in project.resolve().parents)
        # Refuse active FAIR jobs; the request does not authorize killing other jobs.
        active=[]
        for proc in Path('/proc').iterdir():
            if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
            try:
                if proc.stat().st_uid!=os.getuid():continue
                cwd=os.readlink(proc/'cwd');cmd=(proc/'cmdline').read_bytes()
                if cwd==str(target) or cwd.startswith(str(target)+'/') or str(target).encode() in cmd:
                    active.append(int(proc.name))
            except OSError:pass
        out['own_processes_using_target']=active
        assert not active,'active_FAIR_process_detected'
        # Symlinks themselves are removed by rmtree, never their external targets.
        # Refuse nested mounts so the operation stays on the named project tree.
        nested_mounts=[]
        for line in Path('/proc/self/mountinfo').read_text().splitlines():
            mount=Path(line.split(' - ',1)[0].split()[4].replace('\\040',' '))
            if mount==target or target in mount.parents:nested_mounts.append(str(mount))
        out['nested_mounts']=nested_mounts
        assert not nested_mounts,'nested_mount_detected'
        du=subprocess.run(['du','-x','-s','-B1','--',str(target)],capture_output=True,text=True,timeout=40)
        assert du.returncode==0,'size_preflight_incomplete'
        out['target_allocated_bytes_before']=int(du.stdout.split()[0])
        final=target.lstat()
        assert (final.st_dev,final.st_ino)==(initial.st_dev,initial.st_ino)
        assert not target.is_symlink() and target.resolve(strict=True)==target
        out['resolved_absolute_target_verified']=True
        out['deletion_started']=True
        shutil.rmtree(target)
        assert not os.path.lexists(target)
        out['status']='AUTHORIZED_FAIR_PROJECT_DELETED'
except Exception as exc:
    out['status']='DELETION_FAILED_OR_BLOCKED'
    out['error_type']=type(exc).__name__
    out['error_reason']=str(exc)[:500]
out['target_exists_after']=os.path.lexists(target)
out['disk_free_after_bytes']=shutil.disk_usage(project).free
out['observed_free_gain_bytes']=out['disk_free_after_bytes']-out['disk_free_before_bytes']
out['wall_seconds']=time.monotonic()-started
print(json.dumps(out))
if out['status']=='DELETION_FAILED_OR_BLOCKED':raise SystemExit(2)
