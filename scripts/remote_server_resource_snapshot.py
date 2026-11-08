"""Read-only system/process resource snapshot; no torch, allocation or job edits."""
from pathlib import Path
from datetime import datetime,timezone
import csv,json,os,resource,shutil,subprocess,time

started=time.monotonic()
allowed_before=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{allowed_before[0]})
resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
root=Path.home()/'rsg-former'
pointer_path=root/'reports/VG_ACTIVE_EXPERIMENT.json'
pointer=json.loads(pointer_path.read_text()) if pointer_path.exists() else {'artifact_root':'artifacts/vg_method_v2'}
pid_paths={'vg_training':root/pointer['artifact_root']/'pipeline.pid',
           'oi_pixel_worker':root/'artifacts/caption_readiness/oi_stream_phase_v1/pipeline.pid'}
pids={label:int(path.read_text().strip()) for label,path in pid_paths.items() if path.exists()}

def cpu_rows():
    rows={}
    for line in Path('/proc/stat').read_text().splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu'):
            numbers=list(map(int,fields[1:9]));rows[fields[0]]=numbers
    return rows

def process(pid):
    try:
        base=Path('/proc')/str(pid)
        fields=(base/'stat').read_text().rsplit(')',1)[1].split()
        status={line.split(':',1)[0]:line.split(':',1)[1].strip() for line in (base/'status').read_text().splitlines() if ':' in line}
        return {'pid':pid,'active':fields[0]!='Z','state':fields[0],
                'cpu_ticks':int(fields[11])+int(fields[12]),'start_ticks':int(fields[19]),
                'rss_bytes':int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                'peak_rss_bytes':int(status.get('VmHWM','0 kB').split()[0])*1024,
                'threads':int(fields[17]),'cpu_affinity':sorted(os.sched_getaffinity(pid))}
    except (FileNotFoundError,ProcessLookupError):return {'pid':pid,'active':False}
    except PermissionError:return {'pid':pid,'read_permission_denied':True}

first=cpu_rows();proc_first={label:process(pid) for label,pid in pids.items()};sample_start=time.monotonic()
time.sleep(1)
second=cpu_rows();elapsed=time.monotonic()-sample_start
percent={}
for label,values in second.items():
    delta=[b-a for a,b in zip(first.get(label,values),values)];total=sum(delta)
    percent[label]={'busy_percent':100*(total-delta[3]-delta[4])/total if total else 0,
                    'iowait_percent':100*delta[4]/total if total else 0}
processes={}
for label,pid in pids.items():
    after=process(pid);before=proc_first[label]
    if after.get('active') and before.get('active') and after['start_ticks']==before['start_ticks']:
        after['cpu_percent_one_core_units']=100*(after['cpu_ticks']-before['cpu_ticks'])/os.sysconf('SC_CLK_TCK')/elapsed
    after.pop('cpu_ticks',None);after.pop('start_ticks',None);processes[label]=after

mem={line.split(':')[0]:int(line.split(':')[1].split()[0])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.endswith('kB')}
model=next((line.split(':',1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')),'unknown')
gpu_fields=['index','name','memory.total','memory.used','memory.free','utilization.gpu','utilization.memory','temperature.gpu','power.draw','power.limit']
gpu_result=subprocess.run(['nvidia-smi','--query-gpu='+','.join(gpu_fields),'--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
gpus=[dict(zip(gpu_fields,[item.strip() for item in row])) for row in csv.reader(gpu_result.stdout.splitlines())] if gpu_result.returncode==0 else []
gpu_process_result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
gpu_processes=[{'pid':row[0].strip(),'used_gpu_memory_MiB':row[1].strip()} for row in csv.reader(gpu_process_result.stdout.splitlines()) if len(row)==2]
disk=[]
for label,path in [('root','/'),('project',str(root)),('tmp','/tmp')]:
    usage=shutil.disk_usage(path);disk.append({'location':label,'device_id':os.stat(path).st_dev,'total_bytes':usage.total,'used_bytes':usage.used,'free_bytes':usage.free})
cgroup={}
for name in ('cpu.max','memory.max','memory.current'):
    path=Path('/sys/fs/cgroup')/name
    if path.exists():cgroup[name]=path.read_text().strip()
out={'utc':datetime.now(timezone.utc).isoformat(),'sample_seconds':elapsed,
     'cpu':{'model':model,'logical_cores':os.cpu_count(),'initial_probe_allowed_cores':len(allowed_before),
            'load_average_1_5_15':os.getloadavg(),'system':percent['cpu'],
            'busiest_cores':sorted([(k,v) for k,v in percent.items() if k!='cpu'],key=lambda item:item[1]['busy_percent'],reverse=True)[:8]},
     'ram':{key:mem.get(key) for key in ['MemTotal','MemAvailable','MemFree','SwapTotal','SwapFree']},
     'gpus':gpus,'gpu_query_returncode':gpu_result.returncode,
     'gpu_processes':gpu_processes,'processes':processes,'disk':disk,'cgroup':cgroup,
     'measurement_notes':['CPU sampled over1second;100percent process CPU means one fully utilized core','GPU memory unit MiB, GPU power unit W, temperature Celsius','System metrics include other jobs; own process metrics are separate','Repeated disk device IDs refer to the same filesystem'],
     'probe_cpu_affinity_count':1,'probe_address_space_cap_bytes':256*1024**2,
     'live_jobs_modified':False,'gpu_model_allocated':False,'wall_seconds':time.monotonic()-started}
print(json.dumps(out))
