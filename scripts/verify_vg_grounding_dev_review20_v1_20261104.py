"""Verify original JPEGs, fixed20 data, blank annotation and offline HTML UI."""
from pathlib import Path
from datetime import datetime,timezone
import csv,ctypes,hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 result=json.loads((ROOT/'reports/VG_GROUNDING_DEV_REVIEW20_RESULT_20261104.json').read_text());cfg=json.loads((ROOT/'configs/vg_grounding_dev_review20_v1_20261104.json').read_text());pr=json.loads((ROOT/'reports/VG_GROUNDING_DEV_REVIEW20_PREREGISTRATION_20261104.json').read_text())
 assert sha(ROOT/'scripts/build_vg_grounding_dev_review20_v1_20261104.py')==pr['source_sha256'] and sha(ROOT/'configs/vg_grounding_dev_review20_v1_20261104.json')==pr['config_sha256']
 for rel,h in pr['input_sha256'].items():assert sha(ROOT/rel)==h
 out=ROOT/result['local_output_directory'];artifact=ROOT/cfg['archive_directory'];archive=artifact/'source_images.zip';assert sha(archive)==result['source_images_zip_sha256']
 for rel,h in result['output_sha256'].items():assert sha(out/rel)==h
 packet=json.loads((out/'review_data.json').read_text());original=json.loads((ROOT/cfg['packet']).read_text());assert len(packet['units'])==20 and [r['unit_id'] for r in packet['units']]==cfg['fixed_unit_ids']
 with zipfile.ZipFile(archive) as z:
  for row in json.loads((out/'source_manifest.json').read_text())['images']:
   f=out/'images'/(str(row['image_id'])+'.jpg');assert sha(f)==row['jpeg_sha256']==cfg['images'][str(row['image_id'])]['jpeg_sha256'];assert f.read_bytes()==z.read('images/'+f.name)
 for row,old in zip(packet['units'],original):
  for k in ['unit_id','outputs','query_graph','references_evaluation_only','raw_selection']:assert row[k]==old[k]
 with (out/'blank_annotation.csv').open(encoding='utf-8-sig') as f:
  rows=list(csv.DictReader(f));assert len(rows)==20
  for row in rows:assert all(not value for k,value in row.items() if k not in ['unit_id','parent_image_id'])
 html=(out/'index.html').read_text(encoding='utf8');payload=html.split('const DATA=',1)[1].split(';const NS=',1)[0];assert json.loads(payload)==packet['units']
 assert not any(x in html for x in ['https://','http://cdn','innerHTML'])
 kernel=ctypes.windll.kernel32;kernel.GetCurrentProcess.restype=ctypes.c_void_p;h=kernel.GetCurrentProcess();mask=ctypes.c_size_t();system=ctypes.c_size_t();kernel.GetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_size_t),ctypes.POINTER(ctypes.c_size_t)];kernel.SetProcessAffinityMask.argtypes=[ctypes.c_void_p,ctypes.c_size_t]
 assert kernel.GetProcessAffinityMask(h,ctypes.byref(mask),ctypes.byref(system));bits=[1<<i for i in range(64) if mask.value&(1<<i)];assert kernel.SetProcessAffinityMask(h,sum(bits[:4]))
 from playwright.sync_api import sync_playwright
 errors=[];blocked=[]
 with sync_playwright() as pw:
  browser=pw.chromium.launch(executable_path='C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe',headless=True,args=['--disable-gpu','--disable-background-networking','--disable-component-update','--disable-sync','--no-first-run','--renderer-process-limit=1'])
  page=browser.new_page(viewport={'width':1440,'height':1500},device_scale_factor=1);page.on('pageerror',lambda e:errors.append(str(e)))
  def route(req):
   if req.request.url.startswith(('file:','data:','blob:')):req.continue_()
   else:blocked.append(req.request.url);req.abort()
  page.route('**/*',route);page.goto((out/'index.html').as_uri(),wait_until='load',timeout=20000)
  assert page.locator('.unit').count()==20 and page.locator('.caption').count()==60
  loaded=page.evaluate("async()=>{const urls=[...new Set([...document.querySelectorAll('image')].map(x=>x.getAttribute('href')))];return await Promise.all(urls.map(url=>new Promise(resolve=>{const i=new Image();i.onload=()=>resolve({url,width:i.naturalWidth,height:i.naturalHeight});i.onerror=()=>resolve({url,error:true});i.src=url})));}")
  assert len(loaded)==16 and all(x.get('width',0)>0 and x.get('height',0)>0 for x in loaded)
  first=page.locator('.unit').first;first.locator('.object-list button').first.click();assert first.locator('.node-box.selected').count()==1
  first.locator('details.ref summary').click();assert first.locator('details.ref p').is_visible()
  expected_edges=sum(len(u['query_graph']['edges']) for u in packet['units']);assert page.locator('path.edge').count()==expected_edges
  with page.expect_download() as pending:page.locator('#download').click()
  download=pending.value;download.save_as(str(artifact/'AUTOMATED_UI_TEST_blank_notes_not_human_review.json'))
  notes=json.loads((artifact/'AUTOMATED_UI_TEST_blank_notes_not_human_review.json').read_text());assert len(notes['rows'])==20
  for row in notes['rows']:assert not row['reviewer'] and not row['note'] and all(not row[k] for k in ['reviewed','missing_target','relation_issue','extra_detail','uncertain'])
  page.locator('#jump').select_option('unit-19');assert page.locator('#jump').input_value()=='unit-19'
  page.locator('#jump').select_option('unit-0');page.evaluate('window.scrollTo({top:0,behavior:"instant"})');first.screenshot(path=str(out/'preview_first_region.png'))
  assert page.locator('input[type=checkbox]:checked').count()==0;browser.close()
 assert not errors and not blocked
 verification={'status':'PASS_FIXED20_ORIGINAL_JPEGS_CAPTION_SG_GT_BLANK_LABELS_AND_OFFLINE_UI','utc':datetime.now(timezone.utc).isoformat(),'preregistration_sha256':sha(ROOT/'reports/VG_GROUNDING_DEV_REVIEW20_PREREGISTRATION_20261104.json'),'source_images_zip_sha256':sha(archive),'unit_count':20,'unique_images':16,'captions_rendered':60,'directed_graph_edges_rendered':expected_edges,'original_JPEG_bytes_and_SHA_unchanged':True,'generator_outputs_and_references_unchanged':True,'all_annotation_fields_blank':True,'no_external_network_requests':True,'offline_UI_tests':['20cards/60captions','16JPEGs decoded','node bbox highlight','GT expansion','directed SG edge counts','blank-note JSON export','jump to last/first unit'],'browser_JS_errors':errors,'human_labels_created':0,'visual_semantic_labels_created':0,'preview_source_images_for_UI_only':1,'new_inference_or_metric_scoring':False,'check_or_TEST_opened':False,'local_index':(out/'index.html').relative_to(ROOT).as_posix(),'preview_sha256':sha(out/'preview_first_region.png'),'all_original_result_files_still_match':all(sha(out/rel)==h for rel,h in result['output_sha256'].items())}
 (ROOT/'reports/VG_GROUNDING_DEV_REVIEW20_VERIFICATION_20261104.json').write_text(json.dumps(verification,indent=2),encoding='utf8');print(json.dumps(verification))
if __name__=='__main__':main()
