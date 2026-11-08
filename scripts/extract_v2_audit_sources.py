import json,pathlib,re
root=pathlib.Path(__file__).absolute().parents[1]
r=json.loads((root/'reports/VG_V2_SOURCE_AUDIT.json').read_text())
b=root/'research/references/salience_sgg'
for k,v in r['sources'].items():
 p=b/k;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(v,encoding='utf-8')
print('project_data',r['project_data'])
h=r.get('drive_html','')
print('drive_ids',re.findall(r'data-id="([^"]+)"',h))
print('checkpoint_names',re.findall(r'[^"<>]{0,100}(?:\.ckpt|\.pt)[^"<>]{0,80}',h)[:35])
# Preserve only public file metadata, not the page's transient scripts/nonces.
r.pop('drive_html',None)
(root/'reports/VG_V2_SOURCE_AUDIT.json').write_text(json.dumps(r,indent=2),encoding='utf-8')
