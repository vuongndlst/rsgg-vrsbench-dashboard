from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,shutil,subprocess
os.sched_setaffinity(0,{min(os.sched_getaffinity(0))})
root=Path.home()/'rsg-former'
expected={'scripts/remote_vg_focus_repair_controller_v1_20261028.py': '5662a760e32ba615c27a37f103f6965b1781616d683065bcb683d1c2a605b493', 'src/rsgformer/vg_region_focus_repair_v1.py': '02fd0dd3c1cdafbe92cc78abc4c7ae6f722d932d32da6f8147ca3b4b35a54ee9', 'src/rsgformer/vg_caption_candidate_prompt_v2.py': '069255afd7488c5e9cb6b2e21561614f5413977b5c64583fbc14539bb5b8e60d', 'src/rsgformer/vg_caption_graph_v1.py': '625a7ce696b8b6609de923227bb9184b82d2a5fc96969fd49bdcf7b9891c658e', 'src/rsgformer/qwen_streaming_v1.py': 'a6806b7d5eb0142842af39d9b5ac5995a0cdbe0556619a49cd245d48ddfb8a20', 'src/rsgformer/caption_metrics.py': '1d511787bc935ac65015f2015c64104859ce7691b75f3084361f77ca37c43288', 'src/rsgformer/caption_metrics_v2.py': 'a7aad2184fd65591099874c7169f24d69b7abf0fd8142d8b38c2fa1d4dc71ea2', 'src/rsgformer/caption_metric_runtime.py': 'cfc40111a5e8d28f5b0a18a3c3ac38aee746422545384c8394b67eb8ab03d8b6', 'src/rsgformer/vg_color_execution_common_v1.py': '75b9c5e3e53242958bcb4d470cabaf2a8ffa09ada5d4126331ff1afaeb1ad35b', 'src/rsgformer/numpy_json_adapter_v1.py': '780f0f0ec0c564a0838ef2a59e424db970fe7b6a32dab43fa1df22d41ac5011b', 'configs/qwen_runtime.json': 'a3e4fe99474a9606581bb27bf6270182120fa904e8dfd53b42e54658c12ee65d', 'src/rsgformer/vg_grounding_controls_v1_20261102.py': 'be86e2d61045cc378ee40b82b1c0ef8bead6b99ae6449b742bd4f604bf16f8f6', 'reports/CAPTION_JAVA_PREPARATION.json': '3bd0727e0d27b2a1706b5e6fa20dc7b4b2e1ee1827729eb238f9cd144f46ffe4', 'reports/CAPTION_RUNTIME_PREFLIGHT.json': '5b174339f980e40dbea30d13cd22e5959a842960b8d7e2e6542dbe83899e7468'}
rows={}
for rel,h in expected.items():
 p=root/rel
 actual=hashlib.sha256(p.read_bytes()).hexdigest() if p.is_file() else None
 rows[rel]={'expected':h,'actual':actual,'match':actual==h}
total=0;seen=set()
for p in (root/'artifacts').rglob('*'):
 if p.is_symlink() or not p.is_file():continue
 st=p.stat();key=(st.st_dev,st.st_ino)
 if key not in seen:seen.add(key);total+=st.st_size
free=shutil.disk_usage(root).free;gpu=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])*1024**2
ram=int(next(s for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')).split()[1])*1024
print(json.dumps({'utc':datetime.now(timezone.utc).isoformat(),'existing_dependency_hashes':rows,'existing_dependencies_match':all(v['match'] for v in rows.values()),'disk_free_bytes':free,'aggregate_artifacts_bytes':total,'GPU_free_bytes':gpu,'RAM_available_bytes':ram,'CPU_preflight_admitted':free>=17*1024**3 and total+1024**3<=40*1024**3 and ram>=6*1024**3,'Qwen_admitted':free>=17*1024**3 and total+1024**3<=40*1024**3 and gpu>=38*1024**3 and ram>=14*1024**3,'main_namespace_exists':(root/'artifacts/vg_grounding_train_dev_v1_20261103').exists(),'server_files_written':False}))
