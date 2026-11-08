from pathlib import Path
import json
r=Path.home()/'rsg-former';v=r/'vendor/salience_sgg'
o={'sources':{n:(v/n).read_text() for n in ['models/salience_sgg/deformable_transformer.py','models/dab_deformable_detr/deformable_detr.py','models/dab_deformable_detr/matcher.py','models/salience_sgg/__init__.py','models/dab_deformable_detr/__init__.py','util/misc.py']}}
for n in ['VG_V2_PROGRESS.json','VG_V2_ASSETS.json']:
 p=r/'reports'/n
 if p.exists():o[n]=json.loads(p.read_text())
p=r/'artifacts/vg_method_v2/prepare.log'
if p.exists():o['prepare_log_tail']=p.read_text()[-1800:]
t=json.dumps(o).replace(str(Path.home()),'${REMOTE_HOME}');print(t)
