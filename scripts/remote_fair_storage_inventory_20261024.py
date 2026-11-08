"""Bounded read-only size inventory for the explicitly requested FAIR project."""
from pathlib import Path
from datetime import datetime, timezone
import json, os, resource, shutil, signal, subprocess, time

start=time.monotonic()
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
os.nice(10)
resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
resource.setrlimit(resource.RLIMIT_CPU,(70,70))
signal.alarm(80)
home=Path.home(); fair=home/'fair-project'; project=home/'rsg-former'
out={'utc':datetime.now(timezone.utc).isoformat(),'read_only':True,'files_deleted':0,
     'server_files_written':0,'model_loaded':False,'fair_path':str(fair),
     'exists':fair.exists(),'root_is_symlink':fair.is_symlink(),
     'rsg_filesystem_free_bytes':shutil.disk_usage(project).free,
     'own_processes_using_fair':[]}
if fair.is_dir():
    out['fair_resolved_path']=str(fair.resolve())
    out['same_filesystem_as_rsg']=fair.stat().st_dev==project.stat().st_dev
    result=subprocess.run(['du','-x','-B1','--max-depth=2','--',str(fair)],capture_output=True,text=True,timeout=55)
    sizes=[]
    for line in result.stdout.splitlines():
        number,sep,path=line.partition('\t')
        if sep and number.isdigit():sizes.append({'path':path,'allocated_bytes':int(number),'allocated_gib':int(number)/1024**3})
    out['du_exit_code']=result.returncode
    out['complete']=result.returncode==0
    out['du_error_lines']=len(result.stderr.splitlines())
    out['root_allocated_bytes']=next((x['allocated_bytes'] for x in sizes if x['path']==str(fair)),None)
    out['largest_directories']=sorted(sizes,key=lambda x:-x['allocated_bytes'])[:35]
    out['size_semantics']='allocated bytes, du default hardlink dedup, no symlink following, same filesystem only; reclaim is an upper bound until external hardlinks/open files checked'
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            cwd=os.readlink(proc/'cwd');cmd=(proc/'cmdline').read_bytes()
            infair=cwd==str(fair) or cwd.startswith(str(fair)+'/')
            mentions=str(fair).encode() in cmd
            if infair or mentions:out['own_processes_using_fair'].append({'pid':int(proc.name),'cwd_inside_fair':infair,'command_mentions_fair':mentions})
        except OSError:pass
    out['other_users_and_open_file_descriptors_audited']=False
out['wall_seconds']=time.monotonic()-start
print(json.dumps(out))
