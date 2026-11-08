"""Read-only selected Docker labels/name/image/ports; never inspect environment."""
from pathlib import Path
from datetime import datetime,timezone
import json,os,shutil,subprocess
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
ids={'4016943':'50a1fbd79933ca4196d9bfe70ea25f39e28425b9e6e85e78b58ffbf9e04b0408','3578472':'703b4df6df57631ce5e8b49ddda4033ffdfd3c5b337af0d645d705fdebf99da1','2653743':'4ee45fa046addcad290afcd1cf36ebd9927bea9a4af62ff8d0ed55dbac93a825'}
fields=['{{json .Name}}','{{json .Config.Image}}','{{json (index .Config.Labels "com.docker.compose.project")}}','{{json (index .Config.Labels "com.docker.compose.service")}}','{{json (index .Config.Labels "com.docker.compose.project.working_dir")}}','{{json (index .Config.Labels "com.docker.compose.project.config_files")}}','{{json .State.Pid}}','{{json .NetworkSettings.Ports}}']
keys=['name','image','compose_project','compose_service','compose_working_dir','compose_config_files','container_init_host_pid','ports'];rows=[]
for pid,identity in ids.items():
 row={'GPU_pid':int(pid),'container_id':identity}
 if not shutil.which('docker'):row['attribution_status']='docker_cli_unavailable';rows.append(row);continue
 p=subprocess.run(['docker','inspect','--format','|'.join(fields),identity],capture_output=True,text=True,timeout=10)
 if p.returncode:row.update(attribution_status='docker_inspect_not_available_or_not_authorized',error=p.stderr.strip()[:400],returncode=p.returncode)
 else:
  values=p.stdout.strip().split('|');assert len(values)==len(keys)
  row.update({k:json.loads(v) for k,v in zip(keys,values)});row['attribution_status']='name_image_and_selected_compose_labels_read'
 rows.append(row)
print(json.dumps({'status':'READONLY_SELECTED_CONTAINER_PROJECT_METADATA','utc':datetime.now(timezone.utc).isoformat(),'containers':rows,'environment_or_secrets_or_logs_read':False,'container_process_or_GPU_modified':False,'no_sudo_or_privilege_change':True}))
