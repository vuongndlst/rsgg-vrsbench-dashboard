"""Stop only the old fair batch_generate process explicitly authorized by the user."""
from pathlib import Path
from datetime import datetime,timezone
import os,signal,time,json,re
home=Path.home();pid=2916729;p=Path('/proc')/str(pid)
def alias(x):return re.sub(r'/(home|users)/[^/\s]+',r'/\1/${ACCOUNT}',str(x).replace(str(home),'${REMOTE_HOME}').replace(home.name,'${ACCOUNT}'))
before=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
result={'at_utc':datetime.now(timezone.utc).isoformat(),'pid':pid,'authorized_target':'fair batch_generate.py','signals':[]}
if p.exists():
    assert p.stat().st_uid==os.getuid(),'wrong_owner'
    args=[os.fsdecode(x) for x in (p/'cmdline').read_bytes().split(b'\0') if x]
    cwd=Path(os.readlink(p/'cwd'));scripts=[x for x in args if Path(x).name=='batch_generate.py']
    assert scripts and cwd.is_relative_to(home/'fair-project'),'wrong_process_identity'
    ident=(p/'stat').read_text().rsplit(')',1)[1].split()[19]
    result.update(script='batch_generate.py',cwd=alias(cwd),start_ticks=ident)
    # pidfd binds the signal to this exact process, preventing PID-reuse races.
    fd=os.pidfd_open(pid,0)
    assert (p/'stat').read_text().rsplit(')',1)[1].split()[19]==ident
    signal.pidfd_send_signal(fd,signal.SIGTERM,None,0);result['signals'].append('SIGTERM')
    def alive():
        try:
            fields=(p/'stat').read_text().rsplit(')',1)[1].split()
            return fields[19]==ident and fields[0]!='Z'
        except OSError:return False
    for i in range(40):
        if not alive():break
        time.sleep(.25)
    if alive():
        signal.pidfd_send_signal(fd,signal.SIGKILL,None,0);result['signals'].append('SIGKILL')
        for i in range(20):
            if not alive():break
            time.sleep(.25)
    os.close(fd);result['stopped']=not alive()
    assert result['stopped'],'process_did_not_stop'
else:result.update(stopped=True,already_absent=True)
after=os.statvfs(home).f_bavail*os.statvfs(home).f_frsize
result.update(free_before_bytes=before,free_after_bytes=after,observed_free_gain_bytes=after-before,other_jobs_signalled=False)
print(json.dumps(result))
