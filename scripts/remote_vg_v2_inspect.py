from pathlib import Path
import json,subprocess,pwd,os
root=Path.home()/'rsg-former';v=root/'vendor/salience_sgg'
names=['models/salience_sggmodule.py','models/salience_sgg/deformable_detr.py','models/salience_sgg/backbone.py','models/dab_deformable_detrmodule.py','models/dab_deformable_detr/backbone.py','datasets/visual_genome.py','datasets/transforms.py','evaluation/vg_sgg_evaluator.py','models/ops/functions/ms_deform_attn_func.py','models/ops/modules/ms_deform_attn.py','models/__init__.py']
out={'sources':{n:(v/n).read_text() for n in names}}
metadata=root/'data/vg150';out['project_data']=[str(p.relative_to(root)) for p in metadata.rglob('*') if p.is_file()]
import urllib.request
url='https://drive.google.com/drive/folders/1Ge2IjWldzDW3NSv3UFLkmlmzsOkxgHIi'
try:
 html=urllib.request.urlopen(url,timeout=30).read().decode();out['drive_html']=html
except Exception as e:out['drive_error']=type(e).__name__
t=json.dumps(out).replace(str(Path.home()),'${REMOTE_HOME}').replace(pwd.getpwuid(os.getuid()).pw_name,'${ACCOUNT}')
(root/'reports/VG_V2_SOURCE_AUDIT.json').write_text(t);print(t)
